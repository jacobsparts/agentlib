"""
JinaMixin - Mixin that adds Jina AI web tools to any agent.

Provides web_fetch and web_search tools powered by Jina AI's reader and search APIs.
These return LLM-friendly markdown, handling complex HTML, dynamic content, and search results.

Example:
    from agentlib import BaseAgent, JinaMixin

    class MyAgent(JinaMixin, BaseAgent):
        model = 'google/gemini-2.5-flash'
        system = "You are a research assistant."

        @BaseAgent.tool
        def done(self, response: str = "Your response"):
            self.respond(response)

    with MyAgent() as agent:
        result = agent.run("What's the latest news about Python 3.13?")

Configuration:
    - Set JINA_API_KEY env var for higher rate limits (optional, free key at https://jina.ai/?sui=apikey)
    - jina_timeout: Default timeout for requests (default: 60 seconds)

Notes:
    - web_fetch converts any URL to clean markdown
    - web_search performs searches and returns results with content extracted
    - Both support extensive options for filtering, formatting, and localization
"""

import json
import os
import urllib.request
import urllib.error
from typing import Optional

from pydantic import create_model, Field

from .tool_mixin import ToolMixin


class JinaMixin(ToolMixin):
    """Mixin that adds Jina AI web tools. Use with BaseAgent or REPLAgent."""

    # Configuration
    jina_timeout: float = 60.0  # Default timeout for Jina requests

    # === TOOL IMPLEMENTATIONS ===

    def web_fetch(
        self,
        url: str,
        target_selector: Optional[str] = None,
        wait_for_selector: Optional[str] = None,
        remove_selector: Optional[str] = None,
        gather_links: Optional[bool] = None,
        gather_images: Optional[bool] = None,
        generate_image_captions: Optional[bool] = None,
        remove_images: Optional[bool] = None,
        extract_iframes: Optional[bool] = None,
        extract_shadow_dom: Optional[bool] = None,
        return_format: Optional[str] = None,
        link_style: Optional[str] = None,
        engine: Optional[str] = None,
        use_readerlm: Optional[bool] = None,
        token_budget: Optional[int] = None,
        timeout: Optional[int] = None,
        locale: Optional[str] = None,
        proxy_country: Optional[str] = None,
        no_cache: Optional[bool] = None,
        return_json: Optional[bool] = None,
    ) -> str:
        """
        Fetch a URL and return LLM-friendly markdown content.

        Retrieves web page content optimized for LLM consumption. Converts HTML
        to clean markdown, handling complex layouts and dynamic content.

        Args:
            url: URL to fetch
            target_selector: CSS selector to focus on specific elements
            wait_for_selector: CSS selector to wait for before returning
            remove_selector: CSS selector to exclude elements (e.g., 'header,footer,.ads')
            gather_links: Gather all links at end of response
            gather_images: Gather all images at end of response
            generate_image_captions: Generate alt text for images lacking captions
            remove_images: Remove all images from response
            extract_iframes: Include iframe content in response
            extract_shadow_dom: Extract content from Shadow DOM roots
            return_format: Output format: 'markdown', 'html', 'text', 'screenshot', 'pageshot'
            link_style: Link format: 'referenced' (list at end) or 'discarded' (remove)
            engine: Engine: 'browser' (quality), 'direct' (speed), 'cf-browser-rendering' (JS-heavy)
            use_readerlm: Use ReaderLM-v2 for complex HTML structures
            token_budget: Maximum tokens to use for the response
            timeout: Max seconds to wait for page load
            locale: Browser locale for localized content (e.g., 'en-US', 'de-DE')
            proxy_country: Country code for location-based proxy, or 'auto'
            no_cache: Bypass cache for fresh content
            return_json: Return parsed JSON dict instead of markdown

        Returns:
            Markdown content of the page, or error message.
        """
        headers = {}
        api_key = os.environ.get('JINA_API_KEY')
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        if return_json:
            headers['Accept'] = 'application/json'
        if target_selector:
            headers['X-Target-Selector'] = target_selector
        if wait_for_selector:
            headers['X-Wait-For-Selector'] = wait_for_selector
        if remove_selector:
            headers['X-Remove-Selector'] = remove_selector
        if gather_links:
            headers['X-With-Links-Summary'] = 'true'
        if gather_images:
            headers['X-With-Images-Summary'] = 'true'
        if generate_image_captions:
            headers['X-With-Generated-Alt'] = 'true'
        if remove_images:
            headers['X-Retain-Images'] = 'none'
        if extract_iframes:
            headers['X-With-Iframe'] = 'true'
        if extract_shadow_dom:
            headers['X-With-Shadow-Dom'] = 'true'
        if return_format:
            headers['X-Return-Format'] = return_format
        if link_style:
            headers['X-Md-Link-Style'] = link_style
        if engine:
            headers['X-Engine'] = engine
        if use_readerlm:
            headers['X-Respond-With'] = 'readerlm-v2'
        if token_budget:
            headers['X-Token-Budget'] = str(token_budget)
        if timeout:
            headers['X-Timeout'] = str(timeout)
        if locale:
            headers['X-Locale'] = locale
        if proxy_country:
            headers['X-Proxy'] = proxy_country
        if no_cache:
            headers['X-No-Cache'] = 'true'

        jina_url = f'https://r.jina.ai/{url}'
        req = urllib.request.Request(jina_url, headers=headers)

        effective_timeout = timeout or getattr(self, 'jina_timeout', 60.0)
        try:
            with urllib.request.urlopen(req, timeout=effective_timeout) as resp:
                content = resp.read().decode('utf-8')
                if return_json:
                    return json.loads(content)
                return content
        except urllib.error.HTTPError as e:
            return f"[HTTP Error {e.code}] {e.reason}"
        except urllib.error.URLError as e:
            return f"[URL Error] {e.reason}"

    def web_search(
        self,
        query: str,
        site: Optional[str] = None,
        num: Optional[int] = None,
        page: Optional[int] = None,
        country: Optional[str] = None,
        location: Optional[str] = None,
        language: Optional[str] = None,
        gather_links: Optional[bool] = None,
        gather_images: Optional[bool] = None,
        remove_images: Optional[bool] = None,
        return_format: Optional[str] = None,
        engine: Optional[str] = None,
        timeout: Optional[int] = None,
        locale: Optional[str] = None,
        exclude_content: Optional[bool] = None,
        no_cache: Optional[bool] = None,
        return_json: Optional[bool] = None,
    ) -> str:
        """
        Search the web and return LLM-friendly results.

        Performs web search and returns results with page content already
        extracted in markdown format. Good for research and fact-finding.

        Args:
            query: Search query
            site: Limit search to this domain (e.g., 'docs.python.org')
            num: Maximum number of results to return
            page: Result offset for pagination
            country: Two-letter country code for search origin (e.g., 'us', 'de')
            location: City-level location for search origin (e.g., 'Berlin, Germany')
            language: Two-letter language code (e.g., 'en', 'de')
            gather_links: Gather all links at end of response
            gather_images: Gather all images at end of response
            remove_images: Remove all images from response
            return_format: Output format: 'markdown', 'html', 'text', 'screenshot', 'pageshot'
            engine: Engine: 'browser' (quality) or 'direct' (speed)
            timeout: Max seconds to wait for page load
            locale: Browser locale for localized content
            exclude_content: Only return titles/URLs, not page content
            no_cache: Bypass cache for real-time results
            return_json: Return parsed JSON dict instead of markdown

        Returns:
            Search results in markdown format, or error message.
        """
        headers = {'Content-Type': 'application/json'}
        api_key = os.environ.get('JINA_API_KEY')
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        if return_json:
            headers['Accept'] = 'application/json'
        if site:
            headers['X-Site'] = site
        if gather_links:
            headers['X-With-Links-Summary'] = 'true'
        if gather_images:
            headers['X-With-Images-Summary'] = 'true'
        if remove_images:
            headers['X-Retain-Images'] = 'none'
        if return_format:
            headers['X-Return-Format'] = return_format
        if engine:
            headers['X-Engine'] = engine
        if timeout:
            headers['X-Timeout'] = str(timeout)
        if locale:
            headers['X-Locale'] = locale
        if exclude_content:
            headers['X-Respond-With'] = 'no-content'
        if no_cache:
            headers['X-No-Cache'] = 'true'

        # Body params per API spec
        body = {'q': query}
        if num:
            body['num'] = num
        if page:
            body['page'] = page
        if country:
            body['gl'] = country
        if location:
            body['location'] = location
        if language:
            body['hl'] = language

        data = json.dumps(body).encode('utf-8')
        req = urllib.request.Request('https://s.jina.ai/', data=data, headers=headers)

        effective_timeout = timeout or getattr(self, 'jina_timeout', 60.0)
        try:
            with urllib.request.urlopen(req, timeout=effective_timeout) as resp:
                content = resp.read().decode('utf-8')
                if return_json:
                    return json.loads(content)
                return content
        except urllib.error.HTTPError as e:
            return f"[HTTP Error {e.code}] {e.reason}"
        except urllib.error.URLError as e:
            return f"[URL Error] {e.reason}"

    # === HOOK IMPLEMENTATIONS ===

    def _build_system_prompt(self):
        # Get base system prompt from chain
        if hasattr(super(), '_build_system_prompt'):
            system = super()._build_system_prompt()
        else:
            system = getattr(self, 'system', '')

        # Add web tools instructions
        web_instructions = """

WEB TOOLS:
You have access to web fetching and search tools:
- web_fetch: Retrieve any URL as clean, LLM-friendly markdown
- web_search: Search the web and get results with content extracted

Both tools support extensive options for filtering, formatting, and localization.
Set JINA_API_KEY for higher rate limits."""

        return system + web_instructions

    def _get_dynamic_toolspecs(self):
        # Get specs from chain first
        if hasattr(super(), '_get_dynamic_toolspecs'):
            specs = super()._get_dynamic_toolspecs()
        else:
            specs = {}

        # web_fetch spec
        specs['web_fetch'] = create_model(
            'WebFetch',
            url=(str, Field(..., description="URL to fetch")),
            target_selector=(Optional[str], Field(None, description="CSS selector to focus on specific elements")),
            wait_for_selector=(Optional[str], Field(None, description="CSS selector to wait for before returning")),
            remove_selector=(Optional[str], Field(None, description="CSS selector to exclude elements")),
            gather_links=(Optional[bool], Field(None, description="Gather all links at end of response")),
            gather_images=(Optional[bool], Field(None, description="Gather all images at end of response")),
            remove_images=(Optional[bool], Field(None, description="Remove all images from response")),
            return_format=(Optional[str], Field(None, description="Output format: 'markdown', 'html', 'text'")),
            engine=(Optional[str], Field(None, description="Engine: 'browser', 'direct', or 'cf-browser-rendering'")),
            timeout=(Optional[int], Field(None, description="Max seconds to wait")),
            no_cache=(Optional[bool], Field(None, description="Bypass cache for fresh content")),
            __doc__="Fetch a URL and return LLM-friendly markdown content."
        )

        # web_search spec
        specs['web_search'] = create_model(
            'WebSearch',
            query=(str, Field(..., description="Search query")),
            site=(Optional[str], Field(None, description="Limit search to this domain")),
            num=(Optional[int], Field(None, description="Maximum number of results")),
            country=(Optional[str], Field(None, description="Two-letter country code for search origin")),
            language=(Optional[str], Field(None, description="Two-letter language code")),
            gather_links=(Optional[bool], Field(None, description="Gather all links at end of response")),
            remove_images=(Optional[bool], Field(None, description="Remove all images from response")),
            return_format=(Optional[str], Field(None, description="Output format: 'markdown', 'html', 'text'")),
            timeout=(Optional[int], Field(None, description="Max seconds to wait")),
            exclude_content=(Optional[bool], Field(None, description="Only return titles/URLs, not page content")),
            no_cache=(Optional[bool], Field(None, description="Bypass cache for real-time results")),
            __doc__="Search the web and return LLM-friendly results with content extracted."
        )

        return specs
