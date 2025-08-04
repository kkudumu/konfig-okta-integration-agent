/**
 * Enhanced DOM utilities inspired by Skyvern
 * Core functions for intelligent element detection and interaction
 */

// Enhanced element visibility detection
function isElementVisible(element) {
  if (element.tagName.toLowerCase() === "option" ||
      (element.tagName.toLowerCase() === "input" && 
       (element.type === "radio" || element.type === "checkbox"))) {
    return element.parentElement && isElementVisible(element.parentElement);
  }

  const className = element.className ? element.className.toString() : "";
  if (className.includes("select2-offscreen") || 
      className.includes("select2-hidden") || 
      className.includes("ui-select-offscreen")) {
    return false;
  }

  const style = getComputedStyle(element);
  if (!style) return true;
  
  if (style.display === "contents") {
    for (let child = element.firstChild; child; child = child.nextSibling) {
      if (child.nodeType === 1 && isElementVisible(child)) return true;
      if (child.nodeType === 3 && isVisibleTextNode(child)) return true;
    }
    return false;
  }

  if (style.visibility !== "visible") return false;
  
  const rect = element.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return false;

  const center_x = (rect.left + rect.width) / 2 + window.scrollX;
  return center_x >= 0;
}

// Enhanced interactability detection
function isInteractable(element) {
  if (!isElementVisible(element)) return false;
  if (element.hidden) return false;
  
  const tagName = element.tagName.toLowerCase();
  const style = getComputedStyle(element);
  
  // Check cursor pointer
  if (style?.cursor === "pointer") return true;
  
  // Standard interactable elements
  if (["button", "select", "textarea", "a"].includes(tagName)) return true;
  
  // Input elements (except hidden)
  if (tagName === "input" && element.type !== "hidden") return true;
  
  // Elements with click handlers
  if (element.hasAttribute("onclick") || 
      element.hasAttribute("ng-click") ||
      element.hasAttribute("jsaction")) return true;
  
  // ARIA roles
  const role = element.getAttribute("role")?.toLowerCase();
  if (["button", "link", "checkbox", "menuitem", "option"].includes(role)) return true;
  
  return false;
}

// Get element context for better LLM understanding
function getElementContext(element) {
  const contexts = [];
  
  // Check for associated labels
  const elementId = element.getAttribute("id");
  if (elementId) {
    const labels = document.querySelectorAll(`label[for="${elementId}"]`);
    labels.forEach(label => {
      const text = label.textContent?.trim();
      if (text) contexts.push(text);
    });
  }
  
  // Check aria-labelledby and aria-describedby
  const labelledBy = element.getAttribute("aria-labelledby");
  if (labelledBy) {
    const label = document.getElementById(labelledBy);
    if (label) contexts.push(label.textContent?.trim());
  }
  
  const describedBy = element.getAttribute("aria-describedby");
  if (describedBy) {
    const desc = document.getElementById(describedBy);
    if (desc) contexts.push(desc.textContent?.trim());
  }
  
  // Check parent context (fieldset, form, etc.)
  let parent = element.parentElement;
  let depth = 0;
  while (parent && depth < 3) {
    const parentTag = parent.tagName.toLowerCase();
    if (["fieldset", "label", "form"].includes(parentTag)) {
      const parentText = Array.from(parent.childNodes)
        .filter(node => node.nodeType === 3) // Text nodes
        .map(node => node.textContent?.trim())
        .filter(text => text && text.length < 100)
        .join(" ");
      if (parentText) contexts.push(parentText);
      break;
    }
    parent = parent.parentElement;
    depth++;
  }
  
  return contexts.filter(ctx => ctx && ctx.length > 0).join(" | ");
}

// Find all interactable elements with context
function findInteractableElementsWithContext() {
  const elements = [];
  const allElements = document.querySelectorAll("*");
  
  for (const element of allElements) {
    if (isInteractable(element)) {
      const rect = element.getBoundingClientRect();
      const context = getElementContext(element);
      const text = element.textContent?.trim() || "";
      
      elements.push({
        element: element,
        tagName: element.tagName.toLowerCase(),
        type: element.type || "",
        text: text.substring(0, 200), // Limit text length
        context: context,
        rect: {
          x: rect.x,
          y: rect.y,
          width: rect.width,
          height: rect.height
        },
        selector: generateSelector(element),
        attributes: getRelevantAttributes(element)
      });
    }
  }
  
  return elements;
}

// Generate a reliable selector for an element
function generateSelector(element) {
  // Try ID first
  if (element.id) {
    return `#${element.id}`;
  }
  
  // Try unique attribute combinations
  const tagName = element.tagName.toLowerCase();
  const type = element.type;
  const name = element.name;
  const className = element.className;
  
  if (name) {
    return `${tagName}[name="${name}"]`;
  }
  
  if (type && ["submit", "button", "email", "password"].includes(type)) {
    return `${tagName}[type="${type}"]`;
  }
  
  // Try text content for buttons and links
  if (["button", "a"].includes(tagName)) {
    const text = element.textContent?.trim();
    if (text && text.length < 50) {
      return `${tagName}:has-text("${text}")`;
    }
  }
  
  // Fallback to class-based selector
  if (className && typeof className === 'string') {
    const classes = className.split(' ').filter(c => c.length > 0);
    if (classes.length > 0) {
      return `${tagName}.${classes[0]}`;
    }
  }
  
  // Last resort: use xpath-style nth-child
  return generateXPathSelector(element);
}

// Generate XPath-style selector
function generateXPathSelector(element) {
  const path = [];
  let current = element;
  
  while (current && current !== document.body) {
    const tagName = current.tagName.toLowerCase();
    const siblings = Array.from(current.parentElement?.children || [])
      .filter(sibling => sibling.tagName.toLowerCase() === tagName);
    
    if (siblings.length > 1) {
      const index = siblings.indexOf(current) + 1;
      path.unshift(`${tagName}:nth-of-type(${index})`);
    } else {
      path.unshift(tagName);
    }
    
    current = current.parentElement;
  }
  
  return path.join(' > ');
}

// Get relevant attributes for LLM context
function getRelevantAttributes(element) {
  const attrs = {};
  const relevantAttrs = [
    'type', 'name', 'placeholder', 'aria-label', 'title', 
    'role', 'value', 'checked', 'disabled', 'required'
  ];
  
  for (const attr of relevantAttrs) {
    const value = element.getAttribute(attr);
    if (value !== null) {
      attrs[attr] = value;
    }
  }
  
  return attrs;
}

// Utility for text nodes visibility
function isVisibleTextNode(node) {
  const range = node.ownerDocument.createRange();
  range.selectNode(node);
  const rect = range.getBoundingClientRect();
  return rect.width > 0 && rect.height > 0;
}

// Export functions for use in Python
window.skyvernDomUtils = {
  isElementVisible,
  isInteractable,
  getElementContext,
  findInteractableElementsWithContext,
  generateSelector
};