import os
from functools import lru_cache
from typing import Any, cast
from pydantic import BaseModel, Field, model_validator
from langchain.chat_models import init_chat_model
from langchain_anthropic import ChatAnthropic
from deep_research_agent.utils import (
    get_api_key_for_model,
    get_base_url_for_model,
    get_extra_body_for_model,
    get_model_name_for_init,
    get_today_str,
    with_structured_output_for_model,
)
from tests.prompts import RELEVANCE_PROMPT, STRUCTURE_PROMPT, GROUNDEDNESS_PROMPT, OVERALL_QUALITY_PROMPT, CORRECTNESS_PROMPT, COMPLETENESS_PROMPT

DEFAULT_EVAL_MODEL = "deepseek:deepseek-v4-flash"

def _clamp_score(value: Any, default: int = 3) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        score = default
    return max(1, min(5, score))

def _compact_reasoning(data: Any) -> str:
    if isinstance(data, str):
        return data
    return str(data)

def _average_score(values: list[Any], default: int = 3) -> int:
    numeric_values = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("score") or value.get("overall_section_relevance")
        try:
            numeric_values.append(float(value))
        except (TypeError, ValueError):
            pass
    if not numeric_values:
        return default
    return _clamp_score(sum(numeric_values) / len(numeric_values), default=default)

def _get_score_by_alias(data: dict[str, Any], *aliases: str):
    normalized = {
        key.lower().replace(" ", "_").replace("-", "_"): value
        for key, value in data.items()
    }
    for alias in aliases:
        value = normalized.get(alias.lower().replace(" ", "_").replace("-", "_"))
        if value is not None:
            return value
    return None

def _get_eval_model_name() -> str:
    return os.getenv("EVAL_MODEL") or os.getenv("DRA_EVAL_MODEL") or DEFAULT_EVAL_MODEL

@lru_cache(maxsize=1)
def _get_eval_model():
    model_name = _get_eval_model_name()
    kwargs = {
        "model": get_model_name_for_init(model_name),
        "api_key": get_api_key_for_model(model_name, {}),
        "base_url": get_base_url_for_model(model_name),
        "extra_body": get_extra_body_for_model(model_name),
    }
    max_tokens = os.getenv("EVAL_MODEL_MAX_TOKENS") or os.getenv("DRA_EVAL_MODEL_MAX_TOKENS")
    if max_tokens:
        kwargs["max_tokens"] = int(max_tokens)
    return init_chat_model(**kwargs)

def _with_structured_output(schema):
    return with_structured_output_for_model(_get_eval_model(), schema, _get_eval_model_name())

def _json_output_instruction(schema) -> str:
    if schema is OverallQualityScore:
        shape = """{
  "research_depth": 1,
  "source_quality": 1,
  "analytical_rigor": 1,
  "practical_value": 1,
  "balance_and_objectivity": 1,
  "writing_quality": 1
}"""
    elif schema is GroundednessScore:
        shape = """{
  "claims": [
    {
      "claim": "string",
      "grounded": true
    }
  ]
}"""
    else:
        shape = """{
  "reasoning": "string",
  "score": 1
}"""

    return f"""Return only valid JSON. Do not include Markdown, headings, tables, code fences, XML tags, or explanatory text before or after the JSON.
The JSON must match this shape exactly:
{shape}
All score fields must be integers from 1 to 5."""

def _invoke_structured(schema, messages: list[dict]):
    messages = [
        *messages,
        {"role": "user", "content": _json_output_instruction(schema)},
    ]
    return _with_structured_output(schema).invoke(messages)

def _format_for_eval_provider(content: str):
    if isinstance(_get_eval_model(), ChatAnthropic):
        return [{
            "type": "text",
            "text": content,
            "cache_control": {"type": "ephemeral", "ttl": "1h"}
        }]
    return content

def _format_input_query(inputs: dict) -> str:
    messages = inputs["messages"]
    if len(messages) == 1:
        return messages[0]["content"]

    role_to_string_format_map = {
        "user": "<user_input>\n{content}\n</user_input>",
        "assistant": "<assistant_follow_up>\n{content}\n</assistant_follow_up>",
    }

    return "\n\n".join([role_to_string_format_map[message["role"]].format(content=message["content"]) for message in messages])


class OverallQualityScore(BaseModel):
    """Score the overall quality of the report against specific criteria."""
    research_depth: int = Field(description="Integer score 1-5 showing whether the report meets the provided criteria (1 = doesn't meet at all, 5 = meets all criteria).")
    source_quality: int = Field(description="Integer score 1-5 showing whether the report meets the provided criteria (1 = doesn't meet at all, 5 = meets all criteria).")
    analytical_rigor: int = Field(description="Integer score 1-5 showing whether the report meets the provided criteria (1 = doesn't meet at all, 5 = meets all criteria).")
    practical_value: int = Field(description="Integer score 1-5 showing whether the report meets the provided criteria (1 = doesn't meet at all, 5 = meets all criteria).")
    balance_and_objectivity: int = Field(description="Integer score 1-5 showing whether the report meets the provided criteria (1 = doesn't meet at all, 5 = meets all criteria).")
    writing_quality: int = Field(description="Integer score 1-5 showing whether the report meets the provided criteria (1 = doesn't meet at all, 5 = meets all criteria).")

    @model_validator(mode="before")
    @classmethod
    def normalize_provider_output(cls, data: Any):
        if not isinstance(data, dict):
            return data
        dimension_scores = data.get("dimension_scores") or data.get("Dimension Scores")
        if not isinstance(dimension_scores, dict):
            return data
        return {
            **data,
            "research_depth": data.get("research_depth") or _get_score_by_alias(dimension_scores, "research_depth", "research_depth_and_comprehensiveness"),
            "source_quality": data.get("source_quality") or _get_score_by_alias(dimension_scores, "source_quality", "source_quality_and_methodology"),
            "analytical_rigor": data.get("analytical_rigor") or _get_score_by_alias(dimension_scores, "analytical_rigor"),
            "practical_value": data.get("practical_value") or _get_score_by_alias(dimension_scores, "practical_value", "practical_value_and_actionability"),
            "balance_and_objectivity": data.get("balance_and_objectivity") or _get_score_by_alias(dimension_scores, "balance_and_objectivity"),
            "writing_quality": data.get("writing_quality") or _get_score_by_alias(dimension_scores, "writing_quality", "writing_quality_and_clarity"),
        }

def eval_overall_quality(inputs: dict, outputs: dict):
    query = _format_input_query(inputs)
    final_report = outputs["final_report"]
    user_input_content = f"""User input: {query}\n\nReport: \n\n{final_report}\n\nEvaluate whether the report meets the criteria and provide detailed justification for your evaluation."""
    user_input_content = _format_for_eval_provider(user_input_content)
    eval_result = cast(OverallQualityScore, _invoke_structured(OverallQualityScore, [
        {"role": "system", "content": OVERALL_QUALITY_PROMPT.format(today=get_today_str())},
        {"role": "user", "content": user_input_content}
    ]))
    return [
        {"key": "research_depth_score", "score": eval_result.research_depth / 5},
        {"key": "source_quality_score", "score": eval_result.source_quality / 5},
        {"key": "analytical_rigor_score", "score": eval_result.analytical_rigor / 5},
        {"key": "practical_value_score", "score": eval_result.practical_value / 5},
        {"key": "balance_and_objectivity_score", "score": eval_result.balance_and_objectivity / 5},
        {"key": "writing_quality_score", "score": eval_result.writing_quality / 5},
    ]


class RelevanceScore(BaseModel):
    """Score the report relevance against specific criteria."""
    reasoning: str = Field(description="The reason for the score, including specific examples from the report.")
    score: int = Field(description="Integer score 1-5 showing whether the report meets the provided criteria for relevance (1 = doesn't meet at all, 5 = meets all criteria).")

    @model_validator(mode="before")
    @classmethod
    def normalize_provider_output(cls, data: Any):
        if not isinstance(data, dict) or ("reasoning" in data and "score" in data):
            return data
        score = _average_score([
            data.get("topic_relevance"),
            data.get("section_relevance"),
            data.get("citations"),
            data.get("overall_quality"),
        ])
        reasoning = data.get("justification") or data.get("reasoning") or data
        return {**data, "score": score, "reasoning": _compact_reasoning(reasoning)}

def eval_relevance(inputs: dict, outputs: dict):
    query = _format_input_query(inputs)
    final_report = outputs["final_report"]
    user_input_content = f"""User input: {query}\n\nReport: \n\n{final_report}\n\nEvaluate whether the report meets the criteria and provide detailed justification for your evaluation."""
    user_input_content = _format_for_eval_provider(user_input_content)

    eval_result = cast(RelevanceScore, _invoke_structured(RelevanceScore, [
        {"role": "system", "content": RELEVANCE_PROMPT.format(today=get_today_str())},
        {"role": "user", "content": user_input_content}
    ]))
    return {"key": "relevance_score", "score": eval_result.score / 5, "comment": eval_result.reasoning}


class StructureScore(BaseModel):
    """Score the report structure against specific criteria."""
    reasoning: str = Field(description="The reason for the score, including specific examples from the report.")
    score: int = Field(description="Integer score 1-5 showing whether the report meets the provided criteria for structure and flow (1 = doesn't meet at all, 5 = meets all criteria).")

    @model_validator(mode="before")
    @classmethod
    def normalize_provider_output(cls, data: Any):
        if not isinstance(data, dict) or ("reasoning" in data and "score" in data):
            return data
        overall_assessment = data.get("overall_assessment")
        criteria_evaluation = data.get("criteria_evaluation")
        score = overall_assessment.get("score") if isinstance(overall_assessment, dict) else None
        if score is None and isinstance(criteria_evaluation, dict):
            score = _average_score(list(criteria_evaluation.values()))
        reasoning = data.get("reasoning") or overall_assessment or criteria_evaluation or data
        return {**data, "score": _clamp_score(score), "reasoning": _compact_reasoning(reasoning)}

def eval_structure(inputs: dict, outputs: dict):
    query = _format_input_query(inputs)
    final_report = outputs["final_report"]
    user_input_content = STRUCTURE_PROMPT.format(user_question=query, report=final_report, today=get_today_str())
    user_input_content = _format_for_eval_provider(user_input_content)

    eval_result = cast(StructureScore, _invoke_structured(StructureScore, [
        {"role": "user", "content": user_input_content}
    ]))
    return {"key": "structure_and_cohesiveness_score", "score": eval_result.score / 5, "comment": eval_result.reasoning}


class CorrectnessScore(BaseModel):
    """Score the report correctness against specific criteria."""
    reasoning: str = Field(description="The reason for the score, including specific examples from the report.")
    score: int = Field(description="Integer score 1-5 showing whether the report meets the provided criteria for correctness (1 = doesn't meet at all, 5 = meets all criteria).")

def eval_correctness(inputs: dict, outputs: dict, reference_outputs: dict):
    query = _format_input_query(inputs)
    final_report = outputs["final_report"]
    answer = (reference_outputs or {}).get("answer")
    if not answer:
        return {"key": "correctness_score", "score": None, "comment": "Skipped because this dataset example has no reference answer."}
    user_input_content = CORRECTNESS_PROMPT.format(user_question=query, report=final_report, answer=answer, today=get_today_str())
    user_input_content = _format_for_eval_provider(user_input_content)

    eval_result = cast(CorrectnessScore, _invoke_structured(CorrectnessScore, [
        {"role": "user", "content": user_input_content}
    ]))
    return {"key": "correctness_score", "score": eval_result.score / 5, "comment": eval_result.reasoning}

class GroundednessClaim(BaseModel):
    """A claim from the report, and whether or not it is grounded in the context"""
    claim: str = Field(description="The claim extracted from the report.")
    grounded: bool = Field(description="Whether the claim is grounded in the context.")

    @model_validator(mode="before")
    @classmethod
    def normalize_provider_output(cls, data: Any):
        if isinstance(data, dict) and "claim" not in data and "text" in data:
            return {**data, "claim": data["text"]}
        return data

class GroundednessScore(BaseModel):
    """Extract the claims and whether they are grounded in the context"""
    claims: list[GroundednessClaim] = Field(description="All claims extracted from the report, and whether or not they are grounded in the context.")

def eval_groundedness(inputs: dict, outputs: dict):
    final_report = outputs["final_report"]
    context = str(outputs["raw_notes"])

    user_input_content = GROUNDEDNESS_PROMPT.format(context=context, report=final_report, today=get_today_str())
    user_input_content = _format_for_eval_provider(user_input_content)

    eval_result = cast(GroundednessScore, _invoke_structured(GroundednessScore, [
        {"role": "user", "content": user_input_content},
    ]))
    # normalize to 0-1
    if not eval_result.claims:
        return {"key": "groundedness_score", "score": None, "comment": "Skipped because the evaluator returned no claims."}
    grounded_claims = [claim for claim in eval_result.claims if claim.grounded]
    return {"key": "groundedness_score", "score": len(grounded_claims) / len(eval_result.claims), "comment": str(eval_result.claims)}


class CompletenessScore(BaseModel):
    """Score the report completeness against specific criteria."""
    reasoning: str = Field(description="The reason for the score, including specific examples from the report.")
    score: int = Field(description="Integer score 1-5 showing whether the report meets the provided criteria for completeness (1 = doesn't meet at all, 5 = meets all criteria).")

    @model_validator(mode="before")
    @classmethod
    def normalize_provider_output(cls, data: Any):
        if not isinstance(data, dict) or ("reasoning" in data and "score" in data):
            return data

        completeness = data.get("completeness")
        if isinstance(completeness, dict):
            uncovered_points = completeness.get("uncovered_points") or []
            covers_user_question = completeness.get("report_covers_user_question")
            covers_research_brief = completeness.get("report_covers_research_brief")
            if not uncovered_points and covers_user_question and covers_research_brief:
                score = 5
            elif not uncovered_points and (covers_user_question or covers_research_brief):
                score = 4
            elif uncovered_points:
                score = 3
            else:
                score = 2
            return {**data, "score": score, "reasoning": _compact_reasoning(completeness)}

        missing_sections = [
            data.get("user_question_points_missing") or [],
            data.get("research_brief_points_missing") or [],
        ]
        missing_count = sum(len(section) for section in missing_sections if isinstance(section, list))
        score = 5 if missing_count == 0 else 4 if missing_count <= 2 else 3
        return {**data, "score": score, "reasoning": _compact_reasoning(data)}

def eval_completeness(inputs: dict, outputs: dict):
    query = _format_input_query(inputs)
    final_report = outputs["final_report"]
    research_brief = outputs["research_brief"]
    user_input_content = COMPLETENESS_PROMPT.format(user_question=query, research_brief=research_brief, report=final_report, today=get_today_str())
    user_input_content = _format_for_eval_provider(user_input_content)

    eval_result = cast(CompletenessScore, _invoke_structured(CompletenessScore, [
        {"role": "user", "content": user_input_content}
    ]))
    return {"key": "completeness_score", "score": eval_result.score / 5, "comment": eval_result.reasoning}
