from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr


class WorkflowTestModel(BaseChatModel):
    use_tool: bool = False
    fail_first_verification: bool = False
    _verification_calls: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "workflow-test-model"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop=None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        system = str(messages[0].content)
        if "Planner Agent" in system:
            message = AIMessage(
                content="""{
                  "objective":"Solve the test problem",
                  "reasoning":"Use one bounded step",
                  "steps":[{
                    "id":"step-1",
                    "description":"Calculate the requested result",
                    "expected_output":"A supported answer",
                    "suggested_tools":["calculator"],
                    "success_criteria":["Correct calculation"],
                    "depends_on":[]
                  }]}
                """
            )
        elif "Solver Agent" in system:
            has_tool_result = any(isinstance(item, ToolMessage) for item in messages)
            if self.use_tool and not has_tool_result:
                message = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "calculator",
                            "args": {"expression": "2+2"},
                            "id": "calculator-call-1",
                            "type": "tool_call",
                        }
                    ],
                )
            else:
                message = AIMessage(content="The verified result is 4.")
        elif "Verifier Agent" in system:
            self._verification_calls += 1
            if self.fail_first_verification and self._verification_calls == 1:
                message = AIMessage(
                    content=(
                        '{"status":"fail","feedback":"Recalculate once",'
                        '"missing_items":["fresh calculation"],'
                        '"replan_required":true,"confidence":0.4}'
                    )
                )
            else:
                message = AIMessage(
                    content=(
                        '{"status":"pass","feedback":"Correct and complete",'
                        '"missing_items":[],"replan_required":false,"confidence":0.95}'
                    )
                )
        elif "Extract only stable facts" in system:
            message = AIMessage(
                content=(
                    '[{"text":"The user prefers concise answers",'
                    '"category":"preference","confidence":1.0}]'
                )
            )
        elif "Update the running conversation summary" in system:
            message = AIMessage(content="The user is testing durable conversation history.")
        else:
            message = AIMessage(content="Test response.")
        return ChatResult(generations=[ChatGeneration(message=message)])
