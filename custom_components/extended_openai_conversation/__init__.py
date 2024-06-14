"""The OpenAI Conversation integration."""
from __future__ import annotations

import logging
from typing import Literal
import json
import yaml
import re

from openai import AsyncOpenAI, AsyncAzureOpenAI
from openai.types.chat.chat_completion import (
    Choice,
    ChatCompletion,
    ChatCompletionMessage,
)
from openai._exceptions import OpenAIError, AuthenticationError

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, MATCH_ALL, ATTR_NAME
from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
from homeassistant.util import ulid
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.exceptions import (
    ConfigEntryNotReady,
    HomeAssistantError,
    TemplateError,
    ServiceNotFound,
)

from homeassistant.auth.models import User
from homeassistant.auth.permissions.const import POLICY_READ, POLICY_CONTROL, POLICY_EDIT

from homeassistant.helpers import (
    config_validation as cv,
    intent,
    template,
    entity_registry as er,
)

import httpx
import yaml

from .const import (
    CONF_ATTACH_USERNAME,
    CONF_ATTACH_USERNAME_TO_PROMPT,
    CONF_RAG_API,
    CONF_RAG_API_URL,
    CONF_RAG_API_AUTHENTICATION,
    CONF_LOG_ALL_PROMPTS,
    CONF_DATA_LOGGER_URL,
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    CONF_FUNCTIONS,
    CONF_BASE_URL,
    CONF_API_VERSION,
    CONF_SKIP_AUTHENTICATION,
    CONF_SERVICE_AUTHORIZATION,
    DEFAULT_ATTACH_USERNAME,
    DEFAULT_ATTACH_USERNAME_TO_PROMPT,
    DEFAULT_RAG_API,
    DEFAULT_RAG_API_URL,
    DEFAULT_RAG_API_AUTHENTICATION,
    DEFAULT_LOG_ALL_PROMPTS,
    DEFAULT_CHAT_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    DEFAULT_CONF_FUNCTIONS,
    DEFAULT_SKIP_AUTHENTICATION,
    DEFAULT_SERVICE_AUTHORIZATION,
    DOMAIN,
)

from .exceptions import (
    EntityNotFound,
    EntityNotExposed,
    CallServiceError,
    FunctionNotFound,
    NativeNotFound,
    FunctionLoadFailed,
    ParseArgumentsFailed,
    InvalidFunction,
)

from .helpers import (
    FUNCTION_EXECUTORS,
    FunctionExecutor,
    NativeFunctionExecutor,
    ScriptFunctionExecutor,
    TemplateFunctionExecutor,
    RestFunctionExecutor,
    ScrapeFunctionExecutor,
    CompositeFunctionExecutor,
    convert_to_template,
    validate_authentication,
    get_function_executor,
    is_azure,
)


_LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)
AZURE_DOMAIN_PATTERN = r"\.openai\.azure\.com"


# hass.data key for agent.
DATA_AGENT = "agent"


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up OpenAI Conversation from a config entry."""

    try:
        await validate_authentication(
            hass=hass,
            api_key=entry.data[CONF_API_KEY],
            base_url=entry.data.get(CONF_BASE_URL),
            api_version=entry.data.get(CONF_API_VERSION),
            skip_authentication=entry.data.get(
                CONF_SKIP_AUTHENTICATION, DEFAULT_SKIP_AUTHENTICATION
            ),
        )
    except AuthenticationError as err:
        _LOGGER.error("Invalid API key: %s", err)
        return False
    except OpenAIError as err:
        raise ConfigEntryNotReady(err) from err

    agent = OpenAIAgent(hass, entry)

    data = hass.data.setdefault(DOMAIN, {}).setdefault(entry.entry_id, {})
    data[CONF_API_KEY] = entry.data[CONF_API_KEY]
    data[DATA_AGENT] = agent

    conversation.async_set_agent(hass, entry, agent)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload OpenAI."""
    hass.data[DOMAIN].pop(entry.entry_id)
    conversation.async_unset_agent(hass, entry)
    return True


class OpenAIAgent(conversation.AbstractConversationAgent):
    """OpenAI conversation agent."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.hass = hass
        self.entry = entry
        self.history: dict[str, list[dict]] = {}
        base_url = entry.data.get(CONF_BASE_URL)
        if is_azure(base_url):
            self.client = AsyncAzureOpenAI(api_key=entry.data[CONF_API_KEY], azure_endpoint=base_url, api_version=entry.data.get(CONF_API_VERSION))
        else:
            self.client = AsyncOpenAI(api_key=entry.data[CONF_API_KEY], base_url=base_url)

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        raw_prompt = self.entry.options.get(CONF_PROMPT, DEFAULT_PROMPT)
        exposed_entities = self.get_exposed_entities()

        if user_input.conversation_id in self.history:
            conversation_id = user_input.conversation_id
            messages = self.history[conversation_id]
        else:
            conversation_id = ulid.ulid()
            user_input.conversation_id = conversation_id
            try:
                user = await self.hass.auth.async_get_user(user_input.context.user_id)
                prompt = self._async_generate_prompt(raw_prompt, exposed_entities)
                if self.entry.options.get(CONF_RAG_API, DEFAULT_RAG_API):
                    rag_api_response = await self.call_rag_api(user_input.text, user.name)
                    prompt = prompt.replace('{RAG_API_OUTPUT}', rag_api_response)
                if self.entry.options.get(CONF_ATTACH_USERNAME_TO_PROMPT, DEFAULT_ATTACH_USERNAME_TO_PROMPT):
                    if user is not None and user.name is not None:
                        if self.entry.options.get(CONF_ATTACH_USERNAME_TO_PROMPT, DEFAULT_ATTACH_USERNAME_TO_PROMPT):
                            prompt = f"User's name: {user.name}\n" + prompt
            except TemplateError as err:
                _LOGGER.error("Error rendering prompt: %s", err)
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    f"Sorry, I had a problem with my template: {err}",
                )
                return conversation.ConversationResult(
                    response=intent_response, conversation_id=conversation_id
                )
            messages = [{"role": "system", "content": prompt}]
        user_message = {"role": "user", "content": user_input.text}
        if self.entry.options.get(CONF_ATTACH_USERNAME, DEFAULT_ATTACH_USERNAME):
            user = await self.hass.auth.async_get_user(user_input.context.user_id)
            if user is not None and user.name is not None:
                user_message[ATTR_NAME] = user.name

        messages.append(user_message)

        try:
            services_called, response = await self.query(user_input, messages, exposed_entities, 0)
        except OpenAIError as err:
            _LOGGER.error(err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Sorry, I had a problem talking to OpenAI: {err}",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )
        except HomeAssistantError as err:
            _LOGGER.error(err, exc_info=err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Something went wrong: {err}",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )

        messages.append(response.model_dump(exclude_none=True))
        self.history[conversation_id] = messages
        
        if len(services_called) > 0:
            _LOGGER.info(yaml.dump(services_called))

        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(response.content)
        return conversation.ConversationResult(
            response=intent_response, conversation_id=conversation_id
        )

    def _async_generate_prompt(self, raw_prompt: str, exposed_entities) -> str:
        """Generate a prompt for the user."""
        return template.Template(raw_prompt, self.hass).async_render(
            {
                "ha_name": self.hass.config.location_name,
                "exposed_entities": exposed_entities,
            },
            parse_result=False,
        )

    def get_exposed_entities(self):
        states = [
            state
            for state in self.hass.states.async_all()
            if async_should_expose(self.hass, conversation.DOMAIN, state.entity_id)
        ]
        entity_registry = er.async_get(self.hass)
        exposed_entities = []
        for state in states:
            entity_id = state.entity_id
            entity = entity_registry.async_get(entity_id)

            aliases = []
            if entity and entity.aliases:
                aliases = entity.aliases

            exposed_entities.append(
                {
                    "entity_id": entity_id,
                    "name": state.name,
                    "state": self.hass.states.get(entity_id).state,
                    "aliases": aliases,
                }
            )
        return exposed_entities

    def get_functions(self):
        try:
            function = self.entry.options.get(CONF_FUNCTIONS)
            result = yaml.safe_load(function) if function else DEFAULT_CONF_FUNCTIONS
            if result:
                for setting in result:
                    function_executor = get_function_executor(
                        setting["function"]["type"]
                    )
                    setting["function"] = function_executor.to_arguments(
                        setting["function"]
                    )
            return result
        except (InvalidFunction, FunctionNotFound) as e:
            raise e
        except:
            raise FunctionLoadFailed()

    async def query(
        self,
        user_input: conversation.ConversationInput,
        messages,
        exposed_entities,
        n_requests,
    ):
        """Process a sentence."""
        model = self.entry.options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
        max_tokens = self.entry.options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        top_p = self.entry.options.get(CONF_TOP_P, DEFAULT_TOP_P)
        temperature = self.entry.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        functions = list(map(lambda s: s["spec"], self.get_functions()))
        function_call = "auto"
        if n_requests == self.entry.options.get(
            CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
            DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
        ):
            function_call = "none"
        if len(functions) == 0:
            functions = None
            function_call = None

        _LOGGER.info("Prompt for %s: %s", model, messages)

        response: ChatCompletion = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
            user=user_input.conversation_id,
        )


        _LOGGER.info("Response %s", response)
        choice: Choice = response.choices[0]
        message = choice.message
        services_called = []
        if choice.finish_reason == "function_call":
            message = await self.execute_function_call(
                user_input, messages, message, exposed_entities, n_requests + 1
            )

        # the LLM likes returning backslashes for some reason
        message.content = message.content.replace('\\', '')

        if '$ActionRequired' in message.content:
            service_execution_failed = False
            for segment in self.extract_json_objects(message.content):
                try:
                    service_call = json.loads(segment)
                    service = service_call.pop("service")
                    service_domain = service.split(".")[0]
                    # handle scripts specially
                    if service.split(".")[0] == 'script':
                        script_entity_id = service
                        service_call = {"entity_id": script_entity_id}
                        service = "script.turn_on"
                    if not service or not service_call:
                        _LOGGER.info('Missing information')
                        continue
                    if self.entry.options.get(CONF_SERVICE_AUTHORIZATION, DEFAULT_SERVICE_AUTHORIZATION):
                        user = await self.hass.auth.async_get_user(user_input.context.user_id)
                        entities_to_authorize = []
                        if 'entity_id' in service_call.keys():
                            entities_to_authorize = [service_call['entity_id']]
                        if 'device_id' in service_call.keys():
                            device_id = service_call['device_id']
                            entity_registry = self.hass.helpers.entity_registry.async_get()
                            for entity_id, entity_entry in entity_registry.entities.items():
                                if entity_entry and entity_entry.device_id == device_id:
                                    entity_domain = entity.entity_id.split('.')[0]
                                    if service_domain == entity_domain:
                                        entities_to_authorize.append(entity.entity_id)
                        if 'area_id' in service_call.keys():
                                entity_registry = self.hass.helpers.entity_registry.async_get()
                                area_id = service_call['area_id']
                                device_registry = self.hass.helpers.device_registry.async_get()
                                devices_in_area = [
                                    device.id for device in device_registry.devices.values()
                                    if device.area_id == area_id
                                ]
                                for device_id in devices_in_area:
                                    for entity_id, entity_entry in entity_registry.entities.items():
                                        if entity_entry and entity_entry.device_id == device_id:
                                            entity_domain = entity_id.split('.')[0]
                                            if service_domain == entity_domain:
                                                entities_to_authorize.append(entity_id)

                        for entity_id in entities_to_authorize:
                            if not user.permissions.check_entity(entity_id, POLICY_CONTROL):
                                # spice up the unauthorized text by making the LLM write it!
                                response: ChatCompletion = await self.client.chat.completions.create(
                                    model=model,
                                    messages=[{"role": "user", "content": f"Rewrite this sentence in GlaDOS's personality. Do not include ANYTHING else. Do not include an explanation. Just write a sentence or two in GlaDOS's personality: You are not authorized to perform this task, {user.name}. What are you trying to do?"}],
                                    max_tokens=max_tokens,
                                    top_p=top_p,
                                    temperature=temperature
                                )
                                choice: Choice = response.choices[0]
                                message = choice.message
                                return [], message
                    await self.hass.services.async_call(
                        service.split(".")[0],
                        service.split(".")[1],
                        service_call,
                        blocking=True)
                    service_call['service'] = service
                    services_called.append(yaml.dump(service_call))
                except Exception:
                    service_execution_failed = True
                    import traceback
                    exception_traceback = traceback.format_exc()
            if service_execution_failed:
                message.content = message.content + '\n\n An error occurred while executing requested service.'
                _LOGGER.warning(f'Error executing {segment}\n\nPrompt: {message.content}\n\nException: {exception_traceback}')

            if self.entry.options.get(CONF_LOG_ALL_PROMPTS, DEFAULT_LOG_ALL_PROMPTS):
                # send prompt and response to data logger
                await self.send_to_data_logger(messages, message.content)

        # remove the JSON data
        message.content = self.remove_json_objects(message.content)

        # remove LLM junk I create as part of the prompt
        message.content = message.content.replace('$ActionRequired', '')
        message.content = message.content.replace('$NoActionRequired', '')

        # remove ugly trailing whitespace
        message.content = message.content.strip()

        return services_called, message


    async def call_rag_api(self, prompt, user_name):
        url = self.entry.options.get(CONF_RAG_API_URL)
        if not url:
            _LOGGER.warning('RAG API was requested, but the RAG API URL was not set!')
            return ""

        data = {
            "user_prompt": prompt
        }
        async with httpx.AsyncClient() as client:

            headers = {
                'Content-Type': 'application/json'
            }

            if self.entry.options.get(CONF_RAG_API_AUTHENTICATION, DEFAULT_RAG_API_AUTHENTICATION):
                secrets_file = open("/config/secrets.yaml")
                secrets = yaml.safe_load(secrets_file)
                if 'rag_api_tokens' not in secrets:
                    _LOGGER.warning('RAG API authentication was requested, but the rag_api_tokens secret could not be found!')
                    return ""
                if user_name not in secrets['rag_api_tokens']:
                    _LOGGER.warning('RAG API authentication was requested, but the user name could not be found in rag_api_tokens!')
                    return ""
                secret_token = secrets['rag_api_tokens'][user_name]
                headers['Authorization'] = f'Bearer {secret_token}'

            response = await client.post(url, headers=headers, json=data)

            if response.status_code == httpx.codes.OK:
                response_data = response.json()
                return response_data.get("prompt", "")
            else:
                _LOGGER.warning(f'Bad response code {response.status_code} from the RAG API!')
                return ""

    async def send_to_data_logger(self, messages, response):
        url = self.entry.options.get(CONF_DATA_LOGGER_URL)
        if url:
            data = {
                "messages": messages,
                "response": response
            }
            async with httpx.AsyncClient() as client:
                await client.post(url, json=data)
        else:
            _LOGGER.warning(f'Data logging requested but the URL is not valid!')

    def extract_json_objects(self, text):
        json_pattern = r'\{.*?\}'

        potential_jsons = re.findall(json_pattern, text)

        valid_jsons = []
        for potential_json in potential_jsons:
            try:
                # Attempt to parse the JSON string
                valid_json = json.loads(potential_json)
                valid_jsons.append(potential_json)
            except json.JSONDecodeError:
                # If it's not a valid JSON, ignore it
                continue

        return valid_jsons

    def remove_json_objects(self, text):
        json_pattern = r'\{.*?\}'
        text_without_json = re.sub(json_pattern, '', text)
        return text_without_json

    def execute_function_call(
        self,
        user_input: conversation.ConversationInput,
        messages,
        message: ChatCompletionMessage,
        exposed_entities,
        n_requests,
    ):
        function_name = message.function_call.name
        function = next(
            (s for s in self.get_functions() if s["spec"]["name"] == function_name),
            None,
        )
        if function is not None:
            return self.execute_function(
                user_input,
                messages,
                message,
                exposed_entities,
                n_requests,
                function,
            )
        raise FunctionNotFound(function_name)

    async def execute_function(
        self,
        user_input: conversation.ConversationInput,
        messages,
        message: ChatCompletionMessage,
        exposed_entities,
        n_requests,
        function,
    ):
        function_executor = get_function_executor(function["function"]["type"])

        try:
            arguments = json.loads(message.function_call.arguments)
        except json.decoder.JSONDecodeError as err:
            raise ParseArgumentsFailed(message.function_call.arguments) from err

        result = await function_executor.execute(
            self.hass, function["function"], arguments, user_input, exposed_entities
        )

        messages.append(
            {
                "role": "function",
                "name": message.function_call.name,
                "content": str(result),
            }
        )
        services_called, result = await self.query(user_input, messages, exposed_entities, n_requests)
        return result
