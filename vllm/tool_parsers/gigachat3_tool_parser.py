# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence

import regex as re

from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.renderers.chat_utils import make_tool_call_id
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser

logger = init_logger(__name__)

REGEX_FUNCTION_CALL = re.compile(
    r"(?:function call<\|role_sep\|>\n|<\|function_call\|>)(.*)",
    re.DOTALL,
)

REGEX_CONTENT_PATTERN = re.compile(
    r"^(.*?)(?:<\|message_sep\|>|<\|function_call\|>)",
    re.DOTALL,
)

NAME_REGEX = re.compile(
    r'"name"\s*:\s*"([^"]*)"',
    re.DOTALL,
)

ARGS_REGEX = re.compile(
    r'"arguments"\s*:\s*(.*)',
    re.DOTALL,
)
ARGS_HEADER_REGEX = re.compile(r'"arguments"\s*:\s*')
FUNC_CALL_HEADER_REGEX = re.compile(
    r"function call<\|role_sep\|>\n|<\|function_call\|>"
)


class GigaChat3ToolParser(ToolParser):
    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        self.tool_started: bool = False
        self.tool_name_sent: bool = False
        self.tool_id: str | None = None
        self.prev_tool_call_arr: list[dict] = []
        self.end_content: bool = False
        self.streamed_args_for_tool: list[str] = []
        self._func_json_start: int = -1
        self._args_start: int = -1
        self._args_scan_pos: int = 0
        self._args_brace_depth: int = 0
        self._args_in_string: bool = False
        self._args_escape: bool = False
        self._args_close_idx: int | None = None
        self._parsed_args_dict: dict | None = None
        self._streamed_args_end: int = -1

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            request.skip_special_tokens = False
        return request

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        function_call = None
        content = None
        if model_output.rstrip().endswith("</s>"):
            model_output = model_output[: model_output.rfind("</s>")]
        m_func = REGEX_FUNCTION_CALL.search(model_output)
        if m_func:
            try:
                function_call = json.loads(m_func.group(1), strict=False)
                if (
                    isinstance(function_call, dict)
                    and "name" in function_call
                    and "arguments" in function_call
                ):
                    if not isinstance(function_call["arguments"], dict):
                        function_call = None
                else:
                    function_call = None
            except json.JSONDecodeError:
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=model_output,
                )
        m_content = REGEX_CONTENT_PATTERN.search(model_output)
        content = m_content.group(1) if m_content else model_output
        if not function_call:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=content if content else None,
            )
        name = function_call["name"]
        args = function_call["arguments"]
        if not isinstance(args, str):
            args = json.dumps(function_call["arguments"], ensure_ascii=False)
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=[
                ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=name,
                        arguments=args,
                    ),
                )
            ],
            content=content if content else None,
        )

    def _advance_args_scan(self, current_text: str, limit: int) -> None:
        if self._args_close_idx is not None:
            return
        i = self._args_scan_pos
        while i < limit:
            ch = current_text[i]
            if self._args_in_string:
                if self._args_escape:
                    self._args_escape = False
                elif ch == "\\":
                    self._args_escape = True
                elif ch == '"':
                    self._args_in_string = False
            else:
                if ch == '"':
                    self._args_in_string = True
                elif ch == "{":
                    self._args_brace_depth += 1
                elif ch == "}":
                    self._args_brace_depth -= 1
                    if self._args_brace_depth == 0:
                        self._args_close_idx = i + 1
                        i += 1
                        break
            i += 1
        self._args_scan_pos = i

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
        if not previous_text or len(current_text) < self._args_scan_pos:
            self._func_json_start = -1
            self._args_start = -1
            self._args_scan_pos = 0
            self._args_brace_depth = 0
            self._args_in_string = False
            self._args_escape = False
            self._args_close_idx = None
            self._parsed_args_dict = None
            self._streamed_args_end = -1

        content = None
        func_name = None
        cur_args = None
        if self._func_json_start == -1:
            search_start = max(0, len(previous_text) - 32)
            m_header = FUNC_CALL_HEADER_REGEX.search(current_text, search_start)
            if m_header is None and search_start > 0:
                m_header = FUNC_CALL_HEADER_REGEX.search(current_text)
            if m_header is not None:
                self._func_json_start = m_header.end()

        if not self.tool_started:
            m_content = REGEX_CONTENT_PATTERN.search(delta_text)
            if m_content:
                content = m_content.group(1)
                self.end_content = True
            else:
                if not self.end_content:
                    content = delta_text
            if self._func_json_start != -1:
                self.tool_started = True
            if content:
                return DeltaMessage(content=content)
        if self._func_json_start == -1:
            return None

        if not self.tool_name_sent:
            name_match = NAME_REGEX.search(current_text, self._func_json_start)
            if name_match:
                func_name = name_match.group(1)

        if self._args_start == -1:
            args_header = ARGS_HEADER_REGEX.search(current_text, self._func_json_start)
            if args_header is not None:
                pos = args_header.end()
                n = len(current_text)
                while pos < n and current_text[pos].isspace():
                    pos += 1
                if pos < n:
                    self._args_start = pos
                    self._args_scan_pos = pos

        args_end = -1
        if self._args_start != -1:
            end = len(current_text)
            while end > self._args_start and current_text[end - 1].isspace():
                end -= 1
            if current_text.endswith("</s>", self._args_start, end):
                end -= 4
                while end > self._args_start and current_text[end - 1].isspace():
                    end -= 1
            self._advance_args_scan(current_text, end)
            if (
                self._args_close_idx is not None
                and end > self._args_close_idx
                and current_text[end - 1] == "}"
            ):
                cand_end = end - 1
                while (
                    cand_end > self._args_start and current_text[cand_end - 1].isspace()
                ):
                    cand_end -= 1
                if cand_end == self._args_close_idx:
                    if self._parsed_args_dict is None:
                        candidate = current_text[self._args_start : cand_end]
                        try:
                            self._parsed_args_dict = json.loads(candidate, strict=False)
                        except json.JSONDecodeError:
                            self._parsed_args_dict = None
                    if self._parsed_args_dict is not None:
                        end = cand_end
                else:
                    candidate = current_text[self._args_start : cand_end]
                    try:
                        self._parsed_args_dict = json.loads(candidate, strict=False)
                        end = cand_end
                    except json.JSONDecodeError:
                        pass
            if end > self._args_start:
                args_end = end

        if not self.prev_tool_call_arr:
            self.prev_tool_call_arr.append({})
        if not self.tool_name_sent:
            if not func_name:
                return None
            self.tool_name_sent = True
            self.tool_id = make_tool_call_id()
            self.prev_tool_call_arr[0]["name"] = func_name
            return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=0,
                        id=self.tool_id,
                        type="function",
                        function=DeltaFunctionCall(
                            name=func_name,
                        ).model_dump(exclude_none=True),
                    )
                ],
            )
        if args_end == -1:
            return None
        if len(self.streamed_args_for_tool) <= 0:
            self.streamed_args_for_tool.append("")
        if self._streamed_args_end == -1:
            delta_args = current_text[self._args_start : args_end]
            if not delta_args:
                return None
            self.streamed_args_for_tool[0] = delta_args
            self._streamed_args_end = args_end
        elif args_end > self._streamed_args_end:
            delta_args = current_text[self._streamed_args_end : args_end]
            self.streamed_args_for_tool[0] += delta_args
            self._streamed_args_end = args_end
        else:
            return None
        cur_args = self.streamed_args_for_tool[0]
        self.prev_tool_call_arr[0]["arguments_str"] = cur_args
        if self._args_close_idx is not None and args_end == self._args_close_idx:
            if self._parsed_args_dict is None:
                try:
                    self._parsed_args_dict = json.loads(cur_args, strict=False)
                except json.JSONDecodeError:
                    self._parsed_args_dict = {}
            self.prev_tool_call_arr[0]["arguments"] = self._parsed_args_dict
        elif self._args_close_idx is not None:
            try:
                self.prev_tool_call_arr[0]["arguments"] = json.loads(
                    cur_args, strict=False
                )
            except json.JSONDecodeError:
                self.prev_tool_call_arr[0]["arguments"] = {}
        else:
            self.prev_tool_call_arr[0]["arguments"] = {}
        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=0,
                    function=DeltaFunctionCall(
                        arguments=delta_args,
                    ).model_dump(exclude_none=True),
                )
            ],
        )
