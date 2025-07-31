"""
Perception Module for Konfig.

This module handles document ingestion, parsing, and knowledge extraction from
vendor documentation to enable the agent to understand SSO configuration requirements.
"""

import asyncio
import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, NavigableString, Tag
from markdownify import markdownify as md

from konfig.config.settings import get_settings
from konfig.modules.memory.memory_module import MemoryModule
from konfig.utils.logging import LoggingMixin


class DocumentSection:
    """Represents a section of parsed documentation."""
    
    def __init__(
        self,
        title: str,
        content: str,
        level: int,
        section_type: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.title = title
        self.content = content
        self.level = level
        self.section_type = section_type
        self.metadata = metadata or {}
        self.subsections: List[DocumentSection] = []
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert section to dictionary representation."""
        return {
            "title": self.title,
            "content": self.content,
            "level": self.level,
            "section_type": self.section_type,
            "metadata": self.metadata,
            "subsections": [s.to_dict() for s in self.subsections]
        }
    
    def get_full_content(self, include_subsections: bool = True) -> str:
        """Get full content including subsections."""
        content = f"## {self.title}\n\n{self.content}"
        
        if include_subsections and self.subsections:
            for subsection in self.subsections:
                content += "\n\n" + subsection.get_full_content()
        
        return content


class PerceptionModule(LoggingMixin):
    """
    Document perception and understanding module.
    
    This module is responsible for:
    - Fetching documentation from URLs
    - Parsing HTML into structured content
    - Identifying SAML-relevant sections
    - Extracting configuration details
    - Creating searchable knowledge chunks
    """
    
    def __init__(self):
        """Initialize the Perception Module."""
        super().__init__()
        self.setup_logging("perception")
        
        self.settings = get_settings()
        self.memory_module = MemoryModule()
        
        # HTTP client for fetching documents
        self.http_client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; Konfig/1.0; +https://konfig.ai)"
            }
        )
        
        # Patterns for identifying SAML-relevant content
        self.saml_patterns = {
            "sso_section": re.compile(
                r"(single sign[- ]?on|sso|saml|authentication|identity provider|idp)",
                re.IGNORECASE
            ),
            "configuration": re.compile(
                r"(configur|setup|implement|integrat|enable|admin|settings)",
                re.IGNORECASE
            ),
            "metadata_url": re.compile(
                r"(metadata[\s-]?url|metadata[\s-]?endpoint|federation[\s-]?metadata)",
                re.IGNORECASE
            ),
            "entity_id": re.compile(
                r"(entity[\s-]?id|issuer|audience|identifier)",
                re.IGNORECASE
            ),
            "acs_url": re.compile(
                r"(acs|assertion[\s-]?consumer|reply[\s-]?url|callback)",
                re.IGNORECASE
            ),
            "certificate": re.compile(
                r"(certificate|cert|x\.?509|public[\s-]?key)",
                re.IGNORECASE
            )
        }
        
        # Selectors and patterns for finding important elements
        self.important_selectors = [
            "input[type='text']",
            "input[type='url']",
            "input[type='email']",
            "textarea",
            "select",
            "button[type='submit']",
            "button:contains('Save')",
            "button:contains('Apply')",
            "[id*='saml']",
            "[id*='sso']",
            "[name*='saml']",
            "[name*='sso']",
            "[class*='saml']",
            "[class*='sso']"
        ]
        
        self.logger.info("Perception Module initialized")
    
    async def process_documentation(
        self,
        url: str,
        job_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process documentation from a URL.
        
        Args:
            url: URL to the documentation
            job_id: Optional job ID for storing knowledge
            
        Returns:
            Processed documentation with structured content
        """
        self.log_method_call("process_documentation", url=url)
        
        try:
            # Fetch the document
            html_content = await self._fetch_document(url)
            
            # Parse and structure the content
            structured_doc = await self._parse_html_document(html_content, url)
            
            # Extract SAML-specific information
            saml_info = await self._extract_saml_information(structured_doc)
            
            # Create knowledge chunks
            chunks = await self._create_knowledge_chunks(structured_doc, saml_info)
            
            # Store in memory if job_id provided
            if job_id:
                await self._store_knowledge(job_id, chunks)
            
            result = {
                "url": url,
                "title": structured_doc.get("title", "Unknown"),
                "vendor_name": self._extract_vendor_name(structured_doc),
                "sections": structured_doc.get("sections", []),
                "saml_info": saml_info,
                "total_chunks": len(chunks),
                "markdown_content": structured_doc.get("markdown_content", ""),
                "relevant_sections": self._identify_relevant_sections(structured_doc)
            }
            
            self.logger.info(
                "Documentation processed",
                url=url,
                vendor=result["vendor_name"],
                num_sections=len(result["sections"])
            )
            
            return result
            
        except Exception as e:
            self.log_error("process_documentation", e, url=url)
            raise
    
    async def _fetch_document(self, url: str) -> str:
        """Fetch HTML content from URL."""
        self.logger.debug("Fetching document", url=url)
        
        try:
            response = await self.http_client.get(url)
            response.raise_for_status()
            
            content_type = response.headers.get("content-type", "").lower()
            if "text/html" not in content_type:
                self.logger.warning("Non-HTML content type", content_type=content_type)
            
            return response.text
            
        except httpx.HTTPError as e:
            self.logger.error("Failed to fetch document", url=url, error=str(e))
            raise
    
    async def _parse_html_document(self, html: str, base_url: str) -> Dict[str, Any]:
        """Parse HTML into structured document."""
        soup = BeautifulSoup(html, "lxml")
        
        # Remove script and style elements
        for element in soup(["script", "style", "noscript"]):
            element.decompose()
        
        # Extract basic metadata
        title = soup.find("title")
        title_text = title.get_text(strip=True) if title else "Unknown Document"
        
        # Find main content area
        main_content = self._find_main_content(soup)
        
        # Parse sections hierarchically
        sections = self._parse_sections(main_content)
        
        # Convert to markdown for better text processing
        markdown_content = self._html_to_markdown(main_content)
        
        # Extract any code examples
        code_examples = self._extract_code_examples(main_content)
        
        # Extract form fields and inputs
        form_fields = self._extract_form_fields(main_content)
        
        return {
            "title": title_text,
            "url": base_url,
            "sections": [s.to_dict() for s in sections],
            "markdown_content": markdown_content,
            "code_examples": code_examples,
            "form_fields": form_fields,
            "raw_text": main_content.get_text(separator="\n", strip=True)
        }
    
    def _find_main_content(self, soup: BeautifulSoup) -> Tag:
        """Find the main content area of the page."""
        # Try common main content selectors
        main_selectors = [
            "main",
            "article",
            "[role='main']",
            "#main-content",
            "#content",
            ".content",
            ".documentation",
            ".docs-content",
            ".main-content"
        ]
        
        for selector in main_selectors:
            main = soup.select_one(selector)
            if main:
                return main
        
        # Fallback to body
        body = soup.find("body")
        if body:
            # Remove navigation, footer, etc.
            for element in body.select("nav, header, footer, aside, .sidebar, .navigation"):
                element.decompose()
            return body
        
        # Last resort: return the whole soup
        return soup
    
    def _parse_sections(self, content: Tag) -> List[DocumentSection]:
        """Parse content into hierarchical sections."""
        sections = []
        current_section = None
        section_stack = []
        
        # Find all headers
        headers = content.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
        
        for header in headers:
            level = int(header.name[1])
            title = header.get_text(strip=True)
            
            # Determine section type based on title
            section_type = self._classify_section(title)
            
            # Collect content until next header
            content_parts = []
            for sibling in header.find_next_siblings():
                if sibling.name and sibling.name.startswith("h"):
                    break
                content_parts.append(str(sibling))
            
            section_content = "\n".join(content_parts)
            section = DocumentSection(title, section_content, level, section_type)
            
            # Build hierarchy
            while section_stack and section_stack[-1].level >= level:
                section_stack.pop()
            
            if section_stack:
                section_stack[-1].subsections.append(section)
            else:
                sections.append(section)
            
            section_stack.append(section)
        
        # If no headers found, create one section with all content
        if not sections and not any(s.subsections for s in sections):
            all_text = content.get_text(separator="\n", strip=True)
            sections.append(DocumentSection("Main Content", all_text, 1))
        
        return sections
    
    def _classify_section(self, title: str) -> str:
        """Classify section type based on title."""
        title_lower = title.lower()
        
        if any(term in title_lower for term in ["prerequisite", "requirement", "before you begin"]):
            return "prerequisites"
        elif any(term in title_lower for term in ["overview", "introduction", "about"]):
            return "overview"
        elif any(term in title_lower for term in ["configure", "setup", "implement", "enable"]):
            return "configuration"
        elif any(term in title_lower for term in ["test", "verify", "troubleshoot"]):
            return "testing"
        elif any(term in title_lower for term in ["example", "sample"]):
            return "example"
        elif self.saml_patterns["sso_section"].search(title):
            return "sso_specific"
        else:
            return "general"
    
    def _html_to_markdown(self, element: Tag) -> str:
        """Convert HTML element to markdown."""
        # Clean up the HTML first
        html_str = str(element)
        
        # Use markdownify for conversion
        markdown = md(
            html_str,
            heading_style="ATX",
            bullets="*",
            code_language="",
            strip=["img", "iframe", "video", "audio"]
        )
        
        # Clean up excessive newlines
        markdown = re.sub(r'\n{3,}', '\n\n', markdown)
        
        return markdown.strip()
    
    def _extract_code_examples(self, content: Tag) -> List[Dict[str, Any]]:
        """Extract code examples from content."""
        examples = []
        
        # Find code blocks
        for code in content.find_all(["code", "pre"]):
            text = code.get_text(strip=True)
            if len(text) > 20:  # Skip tiny snippets
                example = {
                    "code": text,
                    "language": code.get("class", [""])[0].replace("language-", "") if code.get("class") else "unknown",
                    "context": self._get_element_context(code)
                }
                examples.append(example)
        
        return examples
    
    def _extract_form_fields(self, content: Tag) -> List[Dict[str, Any]]:
        """Extract form fields and inputs mentioned in documentation."""
        fields = []
        
        # Look for field descriptions in lists and tables
        for element in content.find_all(["li", "td", "dd"]):
            text = element.get_text(strip=True)
            
            # Check if this describes a form field
            field_patterns = [
                r"(?:field|input|setting|parameter|option)[\s:]+(['\"]?)(.+?)\1",
                r"(['\"]?)(.+?)\1\s*(?:field|input|setting|parameter)",
                r"enter\s+(?:your\s+)?(.+?)(?:\s+in|into|$)",
                r"provide\s+(?:your\s+)?(.+?)(?:\s+in|$)"
            ]
            
            for pattern in field_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    field_name = match.group(2) if match.lastindex >= 2 else match.group(1)
                    field = {
                        "name": field_name.strip(),
                        "description": text,
                        "required": "required" in text.lower() or "must" in text.lower(),
                        "type": self._infer_field_type(field_name, text)
                    }
                    fields.append(field)
                    break
        
        return fields
    
    def _infer_field_type(self, name: str, description: str) -> str:
        """Infer field type from name and description."""
        name_lower = name.lower()
        desc_lower = description.lower()
        
        if any(term in name_lower or term in desc_lower for term in ["url", "endpoint", "uri"]):
            return "url"
        elif any(term in name_lower or term in desc_lower for term in ["email", "mail"]):
            return "email"
        elif any(term in name_lower or term in desc_lower for term in ["password", "secret"]):
            return "password"
        elif any(term in name_lower or term in desc_lower for term in ["certificate", "cert", "pem"]):
            return "textarea"
        elif any(term in name_lower or term in desc_lower for term in ["select", "choose", "option"]):
            return "select"
        else:
            return "text"
    
    def _get_element_context(self, element: Tag, chars_before: int = 100, chars_after: int = 100) -> str:
        """Get text context around an element."""
        # Get previous text
        prev_text = ""
        for prev in element.find_all_previous(text=True, limit=10):
            if isinstance(prev, NavigableString):
                prev_text = prev.strip() + " " + prev_text
                if len(prev_text) >= chars_before:
                    break
        
        # Get following text
        next_text = ""
        for next_elem in element.find_all_next(text=True, limit=10):
            if isinstance(next_elem, NavigableString):
                next_text += " " + next_elem.strip()
                if len(next_text) >= chars_after:
                    break
        
        return (prev_text[-chars_before:] + " [...] " + next_text[:chars_after]).strip()
    
    async def _extract_saml_information(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Extract SAML-specific information from the document."""
        saml_info = {
            "metadata_url": None,
            "entity_id": None,
            "acs_url": None,
            "slo_url": None,
            "certificate_required": False,
            "attributes_required": [],
            "configuration_steps": [],
            "important_notes": []
        }
        
        content = doc.get("markdown_content", "")
        
        # Look for URLs and endpoints
        url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')
        urls = url_pattern.findall(content)
        
        for url in urls:
            url_lower = url.lower()
            if any(term in url_lower for term in ["metadata", "federation", "saml"]):
                saml_info["metadata_url"] = url
            elif any(term in url_lower for term in ["acs", "assertion", "consumer", "callback"]):
                saml_info["acs_url"] = url
            elif any(term in url_lower for term in ["logout", "slo"]):
                saml_info["slo_url"] = url
        
        # Look for entity ID patterns
        entity_patterns = [
            r"entity\s*id[:\s]+([^\s]+)",
            r"issuer[:\s]+([^\s]+)",
            r"audience[:\s]+([^\s]+)",
            r"identifier[:\s]+([^\s]+)"
        ]
        
        for pattern in entity_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                saml_info["entity_id"] = match.group(1).strip()
                break
        
        # Check if certificate is required
        if self.saml_patterns["certificate"].search(content):
            saml_info["certificate_required"] = True
        
        # Extract configuration steps
        saml_info["configuration_steps"] = self._extract_configuration_steps(doc)
        
        # Extract important notes and warnings
        warning_patterns = [
            r"(?:note|important|warning|caution)[:\s]+(.+?)(?:\n|$)",
            r"(?:⚠️|⚡|📌|❗)\s*(.+?)(?:\n|$)"
        ]
        
        for pattern in warning_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            saml_info["important_notes"].extend(matches)
        
        return saml_info
    
    def _extract_configuration_steps(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract configuration steps from documentation."""
        steps = []
        sections = doc.get("sections", [])
        
        # Look for configuration sections
        for section in sections:
            if section.get("section_type") in ["configuration", "sso_specific"]:
                # Extract numbered steps
                content = section.get("content", "")
                
                # Pattern for numbered steps
                step_patterns = [
                    r"(\d+)\.\s+(.+?)(?=\d+\.|$)",
                    r"(?:step\s+)?(\d+)[:\s]+(.+?)(?=step\s+\d+|$)",
                    r"([a-z])\.\s+(.+?)(?=[a-z]\.|$)"
                ]
                
                for pattern in step_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
                    if matches:
                        for i, (num, text) in enumerate(matches):
                            step = {
                                "number": i + 1,
                                "description": text.strip()[:500],  # Limit length
                                "has_url": bool(re.search(r'https?://', text)),
                                "has_field_reference": bool(re.search(r'field|input|button|click|enter|paste', text, re.IGNORECASE))
                            }
                            steps.append(step)
                        break
        
        return steps
    
    async def _create_knowledge_chunks(
        self,
        doc: Dict[str, Any],
        saml_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create knowledge chunks for vector storage."""
        chunks = []
        
        # Chunk the main content
        sections = doc.get("sections", [])
        
        for section in sections:
            # Create chunk for each section
            chunk_text = f"# {section['title']}\n\n{section['content']}"
            
            chunk = {
                "text": chunk_text[:2000],  # Limit chunk size
                "metadata": {
                    "source_url": doc.get("url"),
                    "section_title": section["title"],
                    "section_type": section["section_type"],
                    "section_level": section["level"],
                    "has_saml_info": bool(self.saml_patterns["sso_section"].search(chunk_text))
                },
                "content_type": section["section_type"],
                "chunk_index": len(chunks)
            }
            chunks.append(chunk)
            
            # Also chunk subsections if they exist
            for subsection in section.get("subsections", []):
                subchunk_text = f"## {subsection['title']}\n\n{subsection['content']}"
                subchunk = {
                    "text": subchunk_text[:2000],
                    "metadata": {
                        "source_url": doc.get("url"),
                        "section_title": subsection["title"],
                        "parent_section": section["title"],
                        "section_type": subsection["section_type"]
                    },
                    "content_type": subsection["section_type"],
                    "chunk_index": len(chunks)
                }
                chunks.append(subchunk)
        
        # Create special chunks for SAML information
        if saml_info:
            saml_chunk = {
                "text": f"SAML Configuration Summary:\n{self._format_saml_info(saml_info)}",
                "metadata": {
                    "source_url": doc.get("url"),
                    "content_type": "saml_summary",
                    **saml_info
                },
                "content_type": "saml_summary",
                "chunk_index": len(chunks),
                "relevance_score": 1.0  # High relevance for SAML summary
            }
            chunks.append(saml_chunk)
        
        # Create chunks for code examples
        for example in doc.get("code_examples", []):
            code_chunk = {
                "text": f"Code Example ({example['language']}):\n```\n{example['code']}\n```\nContext: {example['context']}",
                "metadata": {
                    "source_url": doc.get("url"),
                    "language": example["language"],
                    "content_type": "code_example"
                },
                "content_type": "code_example",
                "chunk_index": len(chunks)
            }
            chunks.append(code_chunk)
        
        return chunks
    
    def _format_saml_info(self, saml_info: Dict[str, Any]) -> str:
        """Format SAML information as text."""
        lines = []
        
        if saml_info.get("metadata_url"):
            lines.append(f"Metadata URL: {saml_info['metadata_url']}")
        if saml_info.get("entity_id"):
            lines.append(f"Entity ID: {saml_info['entity_id']}")
        if saml_info.get("acs_url"):
            lines.append(f"ACS URL: {saml_info['acs_url']}")
        if saml_info.get("certificate_required"):
            lines.append("Certificate: Required")
        
        if saml_info.get("configuration_steps"):
            lines.append("\nConfiguration Steps:")
            for step in saml_info["configuration_steps"]:
                lines.append(f"{step['number']}. {step['description']}")
        
        if saml_info.get("important_notes"):
            lines.append("\nImportant Notes:")
            for note in saml_info["important_notes"]:
                lines.append(f"- {note}")
        
        return "\n".join(lines)
    
    async def _store_knowledge(self, job_id: str, chunks: List[Dict[str, Any]]) -> None:
        """Store knowledge chunks in memory."""
        try:
            import uuid
            job_uuid = uuid.UUID(job_id)
            await self.memory_module.store_knowledge_chunks(job_uuid, chunks)
            self.logger.info("Knowledge chunks stored", job_id=job_id, num_chunks=len(chunks))
        except Exception as e:
            self.log_error("_store_knowledge", e, job_id=job_id)
    
    def _extract_vendor_name(self, doc: Dict[str, Any]) -> str:
        """Extract vendor name from documentation."""
        title = doc.get("title", "")
        
        # Common patterns for vendor names in titles
        patterns = [
            r"^(.+?)\s*[\-–|:]\s*(?:SAML|SSO|Single Sign)",
            r"^(?:Configure|Setup|Enable)\s+(.+?)\s+(?:SAML|SSO)",
            r"^(.+?)\s+(?:Documentation|Docs|Guide)",
            r"^(.+?)\s+(?:SAML|SSO)\s+(?:Configuration|Setup)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback: use first part of title
        parts = re.split(r'[\-–|:]', title)
        if parts:
            return parts[0].strip()
        
        return "Unknown Vendor"
    
    def _identify_relevant_sections(self, doc: Dict[str, Any]) -> List[str]:
        """Identify the most relevant sections for SAML configuration."""
        relevant = []
        sections = doc.get("sections", [])
        
        relevance_keywords = [
            "saml", "sso", "single sign", "configuration", "setup",
            "authentication", "identity provider", "idp", "metadata",
            "certificate", "assertion"
        ]
        
        for section in sections:
            title_lower = section["title"].lower()
            content_lower = section.get("content", "").lower()[:500]  # Check first 500 chars
            
            # Calculate relevance score
            score = sum(1 for keyword in relevance_keywords if keyword in title_lower or keyword in content_lower)
            
            if score > 0:
                relevant.append({
                    "title": section["title"],
                    "type": section["section_type"],
                    "score": score
                })
        
        # Sort by relevance score
        relevant.sort(key=lambda x: x["score"], reverse=True)
        
        # Return just the titles of top sections
        return [r["title"] for r in relevant[:5]]
    
    async def close(self):
        """Clean up resources."""
        await self.http_client.aclose()