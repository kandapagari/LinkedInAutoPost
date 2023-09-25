import click
import dotenv
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.schema import (AIMessage, BaseMessage, HumanMessage,
                              SystemMessage)
from rich import print


def load_messages() -> list[BaseMessage]:
    system_prompt = TextLoader('prompts/system.txt').load()[0]
    instructions_prompt = TextLoader('prompts/instructions.txt').load()[0]
    init_prompt = TextLoader('prompts/prompt_init.txt').load()[0]
    messages = [SystemMessage(content=system_prompt.page_content)]
    messages.append(AIMessage(content=instructions_prompt.page_content))
    messages.append(HumanMessage(content=init_prompt.page_content))
    return messages


def main(topic: str = 'AI') -> str:
    _ = dotenv.load_dotenv(dotenv.find_dotenv())
    model = ChatOpenAI(model='gpt-4', temperature=0.2)
    messages = load_messages()
    prompt_template = PromptTemplate.from_file(
        'prompts/prompt_template.txt', input_variables=['topic'])
    prompt = prompt_template.format(topic=topic)
    messages.append(HumanMessage(content=prompt))
    response = model(messages)
    return response.content


@click.command()
@click.argument('topic', default='AI in coding')
def linkedin_post_generate(topic):
    post = main(topic)
    print(f'Generated post for "{topic}": \n{post}')


if __name__ == "__main__":
    linkedin_post_generate()
