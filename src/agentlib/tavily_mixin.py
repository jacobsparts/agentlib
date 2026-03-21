"""
TavilyMixin - Mixin that adds Tavily web tools to any agent.

Provides web_fetch and web_search tools powered by Tavily's extract and search APIs.
Drop-in replacement for JinaMixin with the same tool names.

Example:
    from agentlib import BaseAgent, TavilyMixin

    class MyAgent(TavilyMixin, BaseAgent):
        model = 'google/gemini-2.5-flash'
        system = "You are a research assistant."

        @BaseAgent.tool
        def done(self, response: str = "Your response"):
            self.respond(response)

    with MyAgent() as agent:
        result = agent.run("What's the latest news about Python 3.13?")

Configuration:
    - Set TAVILY_API_KEY env var (required)
    - tavily_timeout: Default timeout for requests (default: 60 seconds)
"""

import json
import os
import urllib.request
import urllib.error
from typing import Optional

from pydantic import create_model, Field

from .tool_mixin import ToolMixin


class TavilyMixin(ToolMixin):
    """Mixin that adds Tavily web tools. Use with BaseAgent or REPLAgent."""

    # Configuration
    tavily_timeout: float = 60.0  # Default timeout for Tavily requests

    # === TOOL IMPLEMENTATIONS ===

    def web_fetch(
        self,
        url: str,
        query: Optional[str] = None,
        chunks_per_source: Optional[int] = None,
        extract_depth: str = "basic",
        include_images: Optional[bool] = None,
        format: str = "markdown",
        timeout: Optional[float] = None,
    ) -> str:
        """
        Fetch a URL and return LLM-friendly content.

        Retrieves web page content optimized for LLM consumption using Tavily's
        extract API. Returns clean markdown or text.

        Args:
            url: URL to fetch
            query: Optional query to rerank content chunks by relevance
            chunks_per_source: Max relevant chunks per source (1-5, only with query)
            extract_depth: Extraction depth: 'basic' or 'advanced'
            include_images: Include extracted image URLs
            format: Output format: 'markdown' or 'text'
            timeout: Max seconds to wait (1.0-60.0)

        Returns:
            Page content in markdown/text format, or error message.
        """
        api_key = os.environ.get('TAVILY_API_KEY')
        if not api_key:
            return "[Error] TAVILY_API_KEY environment variable not set"

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
        }

        body = {'urls': url, 'format': format}
        if query:
            body['query'] = query
        if chunks_per_source is not None:
            body['chunks_per_source'] = chunks_per_source
        if extract_depth != "basic":
            body['extract_depth'] = extract_depth
        if include_images:
            body['include_images'] = True
        if timeout is not None:
            body['timeout'] = timeout

        data = json.dumps(body).encode('utf-8')
        req = urllib.request.Request(
            'https://api.tavily.com/extract',
            data=data,
            headers=headers,
        )

        effective_timeout = timeout or self.tavily_timeout
        try:
            with urllib.request.urlopen(req, timeout=effective_timeout) as resp:
                result = json.loads(resp.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            return f"[HTTP Error {e.code}] {e.reason}"
        except urllib.error.URLError as e:
            return f"[URL Error] {e.reason}"

        # Format results into readable text
        parts = []
        for item in result.get('results', []):
            content = item.get('raw_content', '')
            if content:
                parts.append(content)
        for item in result.get('failed_results', []):
            parts.append(f"[Failed: {item.get('url')}] {item.get('error', 'Unknown error')}")

        return '\n\n'.join(parts) if parts else "[No content extracted]"

    def web_search(
        self,
        query: str,
        search_depth: str = "basic",
        max_results: int = 5,
        topic: Optional[str] = None,
        time_range: Optional[str] = None,
        include_answer: bool = False,
        include_raw_content: bool = False,
        include_domains: Optional[list] = None,
        exclude_domains: Optional[list] = None,
        country: Optional[str] = None,
    ) -> str:
        """
        Search the web and return LLM-friendly results.

        Performs web search and returns results with content extracted.
        Good for research and fact-finding.

        Args:
            query: Search query
            search_depth: Depth: 'basic', 'advanced', 'fast', or 'ultra-fast'
            max_results: Maximum results to return (0-20)
            topic: Search topic: 'general' or 'news'
            time_range: Filter by time: 'day', 'week', 'month', 'year'
            include_answer: Include AI-generated answer summary
            include_raw_content: Include full page content per result
            include_domains: Whitelist of domains to search
            exclude_domains: Blacklist of domains to exclude
            country: Boost results from this country (two-letter code)

        Returns:
            Search results in formatted text, or error message.
        """
        api_key = os.environ.get('TAVILY_API_KEY')
        if not api_key:
            return "[Error] TAVILY_API_KEY environment variable not set"

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
        }

        body = {
            'query': query,
            'search_depth': search_depth,
            'max_results': max_results,
        }
        if topic:
            body['topic'] = topic
        if time_range:
            body['time_range'] = time_range
        if include_answer:
            body['include_answer'] = True
        if include_raw_content:
            body['include_raw_content'] = True
        if include_domains:
            body['include_domains'] = include_domains
        if exclude_domains:
            body['exclude_domains'] = exclude_domains
        if country:
            body['country'] = country

        data = json.dumps(body).encode('utf-8')
        req = urllib.request.Request(
            'https://api.tavily.com/search',
            data=data,
            headers=headers,
        )

        effective_timeout = self.tavily_timeout
        try:
            with urllib.request.urlopen(req, timeout=effective_timeout) as resp:
                result = json.loads(resp.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            return f"[HTTP Error {e.code}] {e.reason}"
        except urllib.error.URLError as e:
            return f"[URL Error] {e.reason}"

        # Format results into readable text
        parts = []

        answer = result.get('answer')
        if answer:
            parts.append(f"Answer: {answer}\n")

        for i, item in enumerate(result.get('results', []), 1):
            title = item.get('title', 'Untitled')
            url = item.get('url', '')
            content = item.get('content', '')
            raw = item.get('raw_content', '')

            entry = f"[{i}] {title}\n{url}"
            if content:
                entry += f"\n{content}"
            if raw:
                entry += f"\n\n--- Full Content ---\n{raw}"
            parts.append(entry)

        return '\n\n'.join(parts) if parts else "[No results found]"

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

Both tools support options for filtering, formatting, and scoping."""

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
            query=(Optional[str], Field(None, description="Optional query to rerank content chunks by relevance")),
            chunks_per_source=(Optional[int], Field(None, description="Max relevant chunks per source (1-5, only with query)")),
            extract_depth=(str, Field("basic", description="Extraction depth: 'basic' or 'advanced'")),
            format=(str, Field("markdown", description="Output format: 'markdown' or 'text'")),
            timeout=(Optional[float], Field(None, description="Max seconds to wait (1.0-60.0)")),
            __doc__="Fetch a URL and return LLM-friendly content."
        )

        # web_search spec
        specs['web_search'] = create_model(
            'WebSearch',
            query=(str, Field(..., description="Search query")),
            search_depth=(str, Field("basic", description="Depth: 'basic', 'advanced', 'fast', or 'ultra-fast'")),
            max_results=(int, Field(5, description="Maximum results to return (0-20)")),
            topic=(Optional[str], Field(None, description="Search topic: 'general' or 'news'")),
            time_range=(Optional[str], Field(None, description="Filter by time: 'day', 'week', 'month', 'year'")),
            include_answer=(bool, Field(False, description="Include AI-generated answer summary")),
            include_raw_content=(bool, Field(False, description="Include full page content per result")),
            include_domains=(Optional[list], Field(None, description="Whitelist of domains to search")),
            exclude_domains=(Optional[list], Field(None, description="Blacklist of domains to exclude")),
            country=(Optional[str], Field(None, description="Boost results from this country (two-letter code)")),
            __doc__="Search the web and return results. Use include_answer=True for a summary."
        )

        return specs
