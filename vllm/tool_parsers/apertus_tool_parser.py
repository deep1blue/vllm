# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tool call parser for Apertus models.

Extracts tool calls from the format:
<|tools_prefix|>[{"function_name": {"arg1": "value1", ...}}, ...]<|tools_suffix|>

Used when --enable-auto-tool-choice --tool-call-parser apertus are set.
"""

import json
from collections.abc import Sequence

import regex as re
from partial_json_parser.core.options import Allow

from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.renderers.chat_utils import make_tool_call_id
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)
from vllm.tool_parsers.utils import (
    find_common_prefix,
    partial_json_loads,
)

logger = init_logger(__name__)

# Apertus special tokens for tool calls
TOOL_CALLS_PREFIX = "<|tools_prefix|>"
TOOL_CALLS_SUFFIX = "<|tools_suffix|>"
_STRIP_TRAILING_CHARS = '}"], '
_RAW_DECODE = json.JSONDecoder().raw_decode


class ApertusToolParser(ToolParser):
    """Tool call parser for Apertus models.

    Handles the extraction of tool calls from text in both non-streaming
    (complete string) and streaming (chunked token) environments.

    The expected Apertus function call format is a JSON array of single-key dictionaries
    sandwiched between special tokens:
    `<|tools_prefix|>[{"function_name": {"arg1": "value1"}}, ...]<|tools_suffix|>`

    Examples:
        >>> tokenizer = ...  # Mock tokenizer
        >>> parser = ApertusToolParser(tokenizer)
        >>> output = 'I will check. <|tools_prefix|>[{"get_weather": '\
            '{"city": "Paris"}}]<|tools_suffix|>'
        >>> request = ChatCompletionRequest(...)
        >>> info = parser.extract_tool_calls(output, request)
        >>> info.content
        "I will check."
        >>> info.tool_calls[0].function.name
        "get_weather"
        >>> info.tool_calls[0].function.arguments
        '{"city": "Paris"}'

    """

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        """Initializes the ApertusToolParser.

        Args:
            tokenizer: The model's tokenizer.
                Must be provided to interact with special tokens.
            tools: Optional list of tools available for the current request.

        Raises:
            ValueError: If the `model_tokenizer`
                is not successfully passed to the base class.

        """
        super().__init__(tokenizer, tools)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )
        # Regex to extract tool calls block (suffix is optional for incomplete outputs)
        self.tool_call_regex = re.compile(
            rf"{re.escape(TOOL_CALLS_PREFIX)}"
            rf"(.*?)"
            rf"(?:{re.escape(TOOL_CALLS_SUFFIX)}|$)",
            re.DOTALL,
        )

        self._reset_streaming_state()

    def _reset_streaming_state(self) -> None:
        """Resets all streaming state variables for a new completion request.

        This clears the delta text buffer and resets the pointers used to
        track the currently streaming tool index and arguments. Called implicitly
        during initialization and should be called between separate streams.
        """
        self.buffered_delta_text = ""
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.streamed_args_for_tool: list[str] = []
        self._last_prefix_idx: int = -1
        self._last_suffix_idx: int = -1
        self._last_text_len: int = 0
        self._active_prefix_idx: int = -1
        self._json_body_start: int = -1
        self._json_scan_pos: int = 0
        self._container_stack: list[str] = []
        self._json_in_string: bool = False
        self._json_escape: bool = False
        self._json_started: bool = False
        self._json_closed_idx: int | None = None
        self._last_sig_char: str = ""
        self._in_string_is_value: bool = False
        self._open_str_tool_id: int = -1
        self._open_str_pending_tail: str | None = None
        self._open_str_json_len: int = 0

    def _advance_json_scan(self, current_text: str, prefix_idx: int) -> None:
        """Incrementally scans the active JSON block after `TOOL_CALLS_PREFIX`."""
        json_start = prefix_idx + len(TOOL_CALLS_PREFIX)
        if (
            prefix_idx != self._active_prefix_idx
            or len(current_text) < self._json_scan_pos
        ):
            self._active_prefix_idx = prefix_idx
            self._json_body_start = -1
            self._json_scan_pos = json_start
            self._container_stack = []
            self._json_in_string = False
            self._json_escape = False
            self._json_started = False
            self._json_closed_idx = None
            self._last_sig_char = ""
            self._in_string_is_value = False
            self._open_str_tool_id = -1
            self._open_str_pending_tail = None
            self._open_str_json_len = 0

        if self._json_closed_idx is not None:
            return

        i = self._json_scan_pos
        n = len(current_text)
        while i < n:
            ch = current_text[i]
            if not self._json_started:
                if ch.isspace():
                    i += 1
                    continue
                self._json_body_start = i
                self._json_started = True

            if self._json_in_string:
                if self._json_escape:
                    self._json_escape = False
                elif ch == "\\":
                    self._json_escape = True
                    self._open_str_pending_tail = None
                elif ch == '"':
                    self._json_in_string = False
                    self._in_string_is_value = False
                    self._open_str_pending_tail = None
                    self._last_sig_char = '"'
            else:
                if ch == '"':
                    self._json_in_string = True
                    self._in_string_is_value = len(self._container_stack) >= 3 and (
                        self._last_sig_char == ":"
                        or (
                            self._container_stack[-1] == "["
                            and self._last_sig_char in ("[", ",")
                        )
                    )
                elif ch == "[":
                    self._container_stack.append("[")
                    self._last_sig_char = "["
                elif ch == "]":
                    if self._container_stack and self._container_stack[-1] == "[":
                        self._container_stack.pop()
                    self._last_sig_char = "]"
                    if not self._container_stack and self._json_body_start >= 0:
                        try:
                            _, idx = _RAW_DECODE(current_text, self._json_body_start)
                            self._json_closed_idx = idx
                            i = idx
                            break
                        except Exception:
                            pass
                elif ch == "{":
                    self._container_stack.append("{")
                    self._last_sig_char = "{"
                elif ch == "}":
                    if self._container_stack and self._container_stack[-1] == "{":
                        self._container_stack.pop()
                    self._last_sig_char = "}"
                    if not self._container_stack and self._json_body_start >= 0:
                        try:
                            _, idx = _RAW_DECODE(current_text, self._json_body_start)
                            self._json_closed_idx = idx
                            i = idx
                            break
                        except Exception:
                            pass
                elif not ch.isspace():
                    self._last_sig_char = ch
            i += 1

        self._json_scan_pos = i

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        """Adjusts the generation request to ensure special tool tokens are not skipped.

        Forces `skip_special_tokens=False` if tools are actively being evaluated,
        ensuring the tools special tokens are surfaced to the engine for parsing.

        Args:
            request: The incoming OpenAI-compatible chat completion request.

        Returns:
            The potentially modified chat completion request.

        """
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            request.skip_special_tokens = False
        return request

    def _buffer_delta_text(self, delta_text: str) -> str:
        """Buffers incoming delta chunks to prevent
        fragmentation of multi-token special tags.

        If a chunk ends with a partial match of
        `<|tools_prefix|>` or `<|tools_suffix|>`,
        it holds that part back until the next chunk clarifies if it's the actual tag
        or just normal text.

        Args:
            delta_text: The newly generated text chunk

        Returns:
            The safe, verified text chunk free of partial tag collisions.

        Examples:
            >>> parser = ApertusToolParser(...)
            >>> parser._buffer_delta_text("Let me check <|tool" \
            "Let me check "  # "<|tool" is buffered internally
            >>> parser._buffer_delta_text("s_prefix|>" \
            "<|tools_prefix|>"  # Buffer released on completion

        """
        self.buffered_delta_text += delta_text
        text = self.buffered_delta_text

        for tag in (TOOL_CALLS_PREFIX, TOOL_CALLS_SUFFIX):
            if text.endswith(tag):
                self.buffered_delta_text = ""
                return text

            # Evaluate longest possible partial match first
            for i in range(len(tag) - 1, 0, -1):
                if text.endswith(tag[:i]):
                    self.buffered_delta_text = text[-i:]
                    return text[:-i]

        self.buffered_delta_text = ""
        return text

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete model response (non-streaming).

        Args:
            model_output: The full completion string generated by the model.
            request: The current chat completion
                request context containing tool schemas.

        Returns:
            An `ExtractedToolCallInformation` object containing normal text content
            and a list of fully formatted `ToolCall` objects.

        Examples:
            >>> output = 'Let me see. <|tools_prefix|>[{"get_weather":' \
                '{"loc": "Paris"}}]<|tools_suffix|>'
            >>> info = parser.extract_tool_calls(output, request)
            >>> info.tools_called
            True
            >>> info.content
            'Let me see.'
            >>> info.tool_calls[0].function.name
            'get_weather'

        """
        match = self.tool_call_regex.search(model_output)
        if not match:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            # group(1) might contain trailing text if the suffix is missing
            matched_text = match.group(1)
            stripped_text = matched_text.lstrip()

            try:
                # Use raw_decode to robustly isolate
                # the valid JSON array from any trailing garbage
                parsed_json, idx = json.JSONDecoder().raw_decode(stripped_text)
                trailing_in_group = stripped_text[idx:]
            except json.JSONDecodeError:
                # Fallback sequentially to partial parser for token-truncated requests
                parsed_json, _ = partial_json_loads(matched_text, Allow.ALL)
                trailing_in_group = ""

            if not isinstance(parsed_json, list):
                parsed_json = [parsed_json] if parsed_json else []

            tool_calls: list[ToolCall] = []
            for obj in parsed_json:
                if isinstance(obj, dict) and obj:
                    name, args = next(iter(obj.items()))
                    tool_calls.append(
                        ToolCall(
                            type="function",
                            id=make_tool_call_id(),
                            function=FunctionCall(
                                name=name,
                                arguments=json.dumps(args, ensure_ascii=False),
                            ),
                        )
                    )

            # Content combines any generated text
            # prior to and safely after the tool block
            content_str = model_output[: match.start()].strip()

            # Surface any hallucinated text inside
            # the regex group (due to missing suffix)
            if trailing_in_group.strip():
                trailing = trailing_in_group.replace(TOOL_CALLS_SUFFIX, "").strip()
                if trailing:
                    content_str = (content_str + "\n" + trailing).strip()

            # Surface text natively generated after the explicit suffix
            after_suffix = (
                model_output[match.end() :].replace(TOOL_CALLS_SUFFIX, "").strip()
            )
            if after_suffix:
                content_str = (content_str + "\n" + after_suffix).strip()

            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content_str if content_str else None,
            )

        except Exception:
            logger.exception("Error extracting tool calls from Apertus response")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        """Handles streaming chunks.

        Args:
            previous_text: The complete model text generated prior to this chunk.
            current_text: The complete model text including this chunk.
            delta_text: The incremental text addition.
            previous_token_ids: Tokens generated prior to this chunk.
            current_token_ids: Total tokens generated.
            delta_token_ids: Incremental token additions.
            request: The chat completion request.

        Returns:
            A `DeltaMessage` with updated content or tool argument diffs, or `None` if
            the chunk shouldn't emit visible changes yet (e.g. it was purely buffered).

        Examples:
            >>> prev = '<|tools_prefix|>[{"get_weather": {"loc'
            >>> cur = '<|tools_prefix|>[{"get_weather": {"location": "Paris"}}'
            >>> delta = 'ation": "Paris"}}'
            >>> msg = parser.extract_tool_calls_streaming(
            ...     prev, cur, delta, ..., request
            ... )
            >>> msg.tool_calls[0].function.arguments
            'ation": "Paris"}'

        """
        delta_text = self._buffer_delta_text(delta_text)
        if not delta_text:
            return None

        if len(current_text) < self._last_text_len or (
            self._last_prefix_idx == -1 and self._last_text_len == 0
        ):
            self._last_prefix_idx = current_text.rfind(TOOL_CALLS_PREFIX)
            self._last_suffix_idx = current_text.rfind(TOOL_CALLS_SUFFIX)
        else:
            search_from = max(0, self._last_text_len - len(TOOL_CALLS_PREFIX))
            if TOOL_CALLS_PREFIX in delta_text or (
                current_text.find(TOOL_CALLS_PREFIX, search_from) != -1
            ):
                self._last_prefix_idx = current_text.rfind(
                    TOOL_CALLS_PREFIX, search_from
                )
            if TOOL_CALLS_SUFFIX in delta_text or (
                current_text.find(TOOL_CALLS_SUFFIX, search_from) != -1
            ):
                self._last_suffix_idx = current_text.rfind(
                    TOOL_CALLS_SUFFIX, search_from
                )
        self._last_text_len = len(current_text)

        # Fast path: normal text generation before any tools are invoked
        if self._last_prefix_idx == -1:
            return DeltaMessage(content=delta_text)

        try:
            return self._extract_streaming(current_text, delta_text)
        except Exception:
            logger.exception("Error in Apertus streaming tool call extraction")
            return None

    def _extract_streaming(
        self, current_text: str, delta_text: str
    ) -> DeltaMessage | None:
        """Core streaming logic.
        Separates visible chat text from JSON blocks and computes diffs.

        Args:
            current_text: The full generated output string so far.
            delta_text: The latest chunk of text added.

        Returns:
            A `DeltaMessage` containing the `content` delta and/or `tool_calls` delta.

        """
        prefix_idx = self._last_prefix_idx
        suffix_idx = self._last_suffix_idx

        is_inside_tools = prefix_idx > suffix_idx

        json_completed = False
        json_end_idx: int | None = None

        # Check if the JSON array successfully closed implicitly
        if is_inside_tools:
            self._advance_json_scan(current_text, prefix_idx)
            if self._json_closed_idx is not None:
                json_end_idx = self._json_closed_idx
                json_completed, is_inside_tools = True, False

        just_finished = (TOOL_CALLS_SUFFIX in delta_text) or json_completed

        # 1. Fast path: Output normal text immediately
        # if we are completely outside tool block constraints
        if not is_inside_tools and not just_finished:
            text = delta_text.replace(TOOL_CALLS_PREFIX, "").replace(
                TOOL_CALLS_SUFFIX, ""
            )
            return DeltaMessage(content=text) if text else None

        # 2. Extract leading and trailing normal text directly adjacent to tool blocks
        content_str = ""
        if TOOL_CALLS_PREFIX in delta_text:
            content_str += delta_text.split(TOOL_CALLS_PREFIX)[0].replace(
                TOOL_CALLS_SUFFIX, ""
            )

        if just_finished:
            if json_completed and json_end_idx is not None:
                # The tool block finished in this chunk via implicit JSON completion
                # Ensure we strictly isolate
                # and extract only trailing text that is part of `delta_text`
                delta_start_idx = len(current_text) - len(delta_text)
                content_start = max(json_end_idx, delta_start_idx)
                if content_start < len(current_text):
                    content_str += current_text[content_start:].replace(
                        TOOL_CALLS_SUFFIX, ""
                    )
            else:
                content_str += delta_text.split(TOOL_CALLS_SUFFIX)[-1]

        # 3. Extract the isolated JSON array string for the active block
        json_start = prefix_idx + len(TOOL_CALLS_PREFIX)
        json_end = (
            suffix_idx
            if suffix_idx > prefix_idx
            else (json_end_idx if json_end_idx is not None else len(current_text))
        )
        json_len = json_end - json_start
        is_final = not is_inside_tools

        if (
            not is_final
            and self.current_tool_id >= 0
            and self.current_tool_name_sent
            and self._open_str_pending_tail is not None
            and self._open_str_tool_id == self.current_tool_id
            and json_len > self._open_str_json_len
        ):
            added = current_text[json_start + self._open_str_json_len : json_end]
            if (
                '"' not in added
                and "\\" not in added
                and all(ord(c) >= 0x20 for c in added)
            ):
                combined = self._open_str_pending_tail + added
                diff = combined.rstrip(_STRIP_TRAILING_CHARS)
                self._open_str_pending_tail = combined[len(diff) :]
                self._open_str_json_len = json_len
                tool_calls = (
                    [
                        DeltaToolCall(
                            index=self.current_tool_id,
                            function=DeltaFunctionCall(arguments=diff).model_dump(
                                exclude_none=True
                            ),
                        )
                    ]
                    if diff
                    else []
                )
                if diff:
                    self.streamed_args_for_tool[self.current_tool_id] += diff
                if tool_calls or content_str:
                    return DeltaMessage(
                        content=content_str if content_str else None,
                        tool_calls=tool_calls if tool_calls else None,
                    )
                return None
            self._open_str_pending_tail = None

        json_str = current_text[json_start:json_end]
        tool_calls = self._parse_and_diff_json(json_str, is_final=is_final)

        if tool_calls or content_str:
            return DeltaMessage(
                content=content_str if content_str else None,
                tool_calls=tool_calls if tool_calls else None,
            )

        return None

    def _parse_and_diff_json(
        self, json_str: str, is_final: bool
    ) -> list[DeltaToolCall]:
        """Parses an isolated, potentially incomplete streaming JSON array and returns
        newly accumulated tool call diffs.

        Args:
            json_str: The extracted JSON array string so far
                (e.g. `[{"weather": {"city": "Par"}]`).
            is_final: True if the tool block has received its closing`<|tools_suffix|>`

        Returns:
            A list of `DeltaToolCall`
            items representing string diffs in function arguments
            to stream back to the client.

        """
        if (
            not is_final
            and self.current_tool_id >= 0
            and self.current_tool_name_sent
            and self._open_str_pending_tail is not None
            and self._open_str_tool_id == self.current_tool_id
            and len(json_str) > self._open_str_json_len
        ):
            added = json_str[self._open_str_json_len :]
            if (
                '"' not in added
                and "\\" not in added
                and all(ord(c) >= 0x20 for c in added)
            ):
                combined = self._open_str_pending_tail + added
                diff = combined.rstrip(_STRIP_TRAILING_CHARS)
                self._open_str_pending_tail = combined[len(diff) :]
                self._open_str_json_len = len(json_str)
                if diff:
                    self.streamed_args_for_tool[self.current_tool_id] += diff
                    return [
                        DeltaToolCall(
                            index=self.current_tool_id,
                            function=DeltaFunctionCall(arguments=diff).model_dump(
                                exclude_none=True
                            ),
                        )
                    ]
                return []
            self._open_str_pending_tail = None

        try:
            # An exponent can change a number's already-streamed representation.
            flags = Allow.ALL if is_final else Allow.ALL & ~Allow.NUM
            parsed, _ = partial_json_loads(json_str, flags)
            if not isinstance(parsed, list):
                parsed = [parsed] if parsed else []
        except Exception:
            return []

        if not parsed:
            return []

        tool_calls: list[DeltaToolCall] = []
        latest_index = len(parsed) - 1

        # Catch up and finalize any tools we fully skipped over in one large text delta
        while self.current_tool_id < latest_index:
            if self.current_tool_id >= 0:
                if not self.current_tool_name_sent:
                    self._emit_tool_name(parsed, self.current_tool_id, tool_calls)

                delta = self._get_tool_diff(
                    parsed, self.current_tool_id, is_final=True, json_str=json_str
                )
                if delta:
                    tool_calls.append(delta)

            self.current_tool_id += 1
            self.current_tool_name_sent = False
            while len(self.streamed_args_for_tool) <= self.current_tool_id:
                self.streamed_args_for_tool.append("")

        # Stream the currently active tool
        if self.current_tool_id >= 0:
            if not self.current_tool_name_sent:
                self._emit_tool_name(parsed, self.current_tool_id, tool_calls)

            delta = self._get_tool_diff(
                parsed, self.current_tool_id, is_final, json_str=json_str
            )
            if delta:
                tool_calls.append(delta)

        return tool_calls

    def _emit_tool_name(
        self, parsed: list, index: int, tool_calls: list[DeltaToolCall]
    ) -> None:
        """Extracts and emits the function name mapped to a new tool call ID.

        Args:
            parsed: The partially parsed JSON list containing tool dictionaries.
            index: The active index within the JSON list.
            tool_calls: The running list of delta chunks to mutate.

        Examples:
            Appends `DeltaToolCall(index=0,
                function=DeltaFunctionCall(name="get_weather", ...))`
            to the `tool_calls` list and marks the name as sent.

        """
        obj = parsed[index]
        if isinstance(obj, dict) and obj:
            name = next(iter(obj))
            self.current_tool_name_sent = True
            tool_calls.append(
                DeltaToolCall(
                    index=index,
                    type="function",
                    id=make_tool_call_id(),
                    function=DeltaFunctionCall(name=name, arguments="").model_dump(
                        exclude_none=True
                    ),
                )
            )

    def _get_tool_diff(
        self,
        parsed: list,
        index: int,
        is_final: bool,
        json_str: str | None = None,
    ) -> DeltaToolCall | None:
        """Calculates the exact string difference to safely append new tool parameters.

        This ensures characters like `{`, `}`, and `"` don't jump around unevenly
        in the UI frontend while streaming incomplete JSON arguments.

        Args:
            parsed: The latest list of parsed JSON objects.
            index: The active tool's array index.
            is_final: Whether to emit
                trailing structural brackets (True if block is done).
            json_str: Optional raw JSON string for incremental open-string state.

        Returns:
            A `DeltaToolCall` mapping to the arguments diff,
                or None if no text was appended.

        Examples:
            >>> # Previous streamed state: '{"city": "Pari'
            >>> # Current full parse state: '{"city": "Paris"}'
            >>> # Returns diff (closing bracket suppressed until final):
            >>> parser._get_tool_diff(parsed, index=0, is_final=False)
            DeltaToolCall(index=0, function=DeltaFunctionCall(arguments='s'))

        """
        obj = parsed[index]
        if not isinstance(obj, dict) or not obj:
            return None

        name, args = next(iter(obj.items()))
        if args is None:
            return None

        args_json_full = json.dumps(args, ensure_ascii=False)
        args_json = (
            args_json_full if is_final else args_json_full.rstrip(_STRIP_TRAILING_CHARS)
        )

        if (
            not is_final
            and json_str is not None
            and self._json_in_string
            and self._in_string_is_value
            and not self._json_escape
            and not json_str.endswith("\\")
        ):
            closing_count = len(self._container_stack) - 1
            if (
                closing_count > 0
                and len(args_json_full) > closing_count
                and len(args_json) > 0
            ):
                prefix_candidate = args_json_full[:-closing_count]
                if prefix_candidate.rstrip(_STRIP_TRAILING_CHARS) == args_json:
                    self._open_str_tool_id = index
                    self._open_str_pending_tail = prefix_candidate[len(args_json) :]
                    self._open_str_json_len = len(json_str)

        prev_sent = self.streamed_args_for_tool[index]
        if args_json == prev_sent:
            return None

        prefix = find_common_prefix(prev_sent, args_json)
        if len(prefix) < len(prev_sent):
            # Backtrack state if partial parser structurally updates a past assumption
            self.streamed_args_for_tool[index] = prefix
            self._open_str_pending_tail = None
            return None

        diff = args_json[len(prev_sent) :]
        if diff:
            self.streamed_args_for_tool[index] = args_json
            return DeltaToolCall(
                index=index,
                function=DeltaFunctionCall(arguments=diff).model_dump(
                    exclude_none=True
                ),
            )

        return None
