import re
from typing import Any

from openai import OpenAI

RE_STRIP = re.compile(r"^[^A-Za-z]+|[^A-Za-z]+$")


def send_prompt(
    *,  # enforce kwargs
    client: OpenAI,
    model: str,
    prompt: str | list[dict[str, str]],
    extra_kwargs: dict[str, Any] = {},
) -> str:
    if model.startswith("o1"):
        kwargs = {
            "model": model,
            "n": 1,
            "max_completion_tokens": 25000,
            "seed": 42,
        }
    else:
        kwargs = {
            "model": model,
            "n": 1,
            "temperature": 0,
            "max_tokens": 1024,
            "seed": 42,
        }
    kwargs |= extra_kwargs

    if isinstance(prompt, list):
        if model.startswith("o1-mini"):
            # As of 2024-12-19, o1-mini does not support "system/developer"
            # messages so replace with "user" role for now
            prompt = [
                {
                    "role": m["role"] if m["role"] != "system" else "user",
                    "content": m["content"],
                }
                for m in prompt
            ]
        elif "Llama-2" in model:
            prompt = merge_role_messages(prompt=prompt, replace_system_with_user=True)
        elif "405b" in model:
            prompt = merge_role_messages(prompt=prompt, replace_system_with_user=False)

        completion = client.chat.completions.create(
            messages=prompt,
            **kwargs,
        )
    else:
        completion = client.completions.create(
            prompt=prompt,
            **kwargs,
        )

    # error if no completion choices
    if len(completion.choices) != 1:
        print(completion)
        return "Model Did Not Return Completion"

    if isinstance(prompt, list):
        output = completion.choices[0].message.content
    else:
        output = completion.choices[0].text

    # possible that this throws error
    return output


def string_extract_answer(
    *,  # enforce kwargs
    output: str,
    answer_delimiter: str,
) -> str:
    parts = output.split(answer_delimiter)
    if len(parts) < 2:
        # TODO log if delimiter not present?
        return "Delimiter Not Found"
    after_delim = parts[-1]
    after_delim = RE_STRIP.sub("", after_delim)
    answers = after_delim.split()
    if len(answers) == 0:
        return ""
    answer = answers[0]
    cleaned = RE_STRIP.sub("", answer)
    return cleaned.lower().title()


def answer_with_constrained_iterative_cot(
    *,  # enforce kwargs
    client: OpenAI,
    model: str,
    prompt: str | list[dict[str, str]],
    output: str,
    answer_delimiter: str,
    answer_choices: list[str],
) -> str:
    # build delimiter into answer choice so chat prompt isn't fragmented
    # e.g. assistant: "Final Answer:", assistant: "Yes"
    choices = [answer_delimiter + " " + c for c in answer_choices]

    parts = output.split(answer_delimiter)
    if len(parts) < 2:
        # TODO log if delimiter not present?
        return "Delimiter Not Found"

    # assumes text preceeding delimiter is "reasoning"
    reasoning = parts[0]
    if isinstance(prompt, list):
        new_prompt = prompt.copy()
        new_prompt.append(
            {
                "role": "assistant",
                "content": reasoning,
            }
        )
    else:
        new_prompt = prompt + reasoning

    output = send_prompt(
        client=client,
        model=model,
        prompt=new_prompt,
        extra_kwargs={
            "extra_body": {
                "guided_choice": choices,  # only works with vLLM backed endpoint
            },
        },
    )
    return output.split(answer_delimiter)[-1].strip()


def merge_role_messages(
    *,  # enforce kwargs
    prompt: list[dict[str, str]],
    replace_system_with_user: bool = False,
) -> list[dict[str, str]]:
    new_prompt = []
    last_msg = None
    for msg in prompt:
        msg = msg.copy()
        if replace_system_with_user and msg["role"] == "system":
            msg["role"] = "user"
        if last_msg is None or msg["role"] != last_msg["role"]:
            new_prompt.append(last_msg)
            last_msg = msg.copy()
        else:
            last_msg["content"] += "\n\n" + msg["content"]
    new_prompt.append(last_msg)
    return new_prompt[1:]
