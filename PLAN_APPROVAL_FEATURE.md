# Plan Approval Feature

## Overview

Added a comprehensive user approval system that shows the complete integration plan before execution, giving users full visibility and control over what the system intends to do.

## Features

### 1. **Interactive Plan Display**
- 📋 **Overview Panel**: Shows vendor, Okta domain, and integration type
- 📊 **Summary Table**: Lists all steps with:
  - Step name and description
  - Category (OKTA, VENDOR, VERIFY, GENERAL)
  - Risk level (LOW, MEDIUM, HIGH)
  - Required vs optional status
  - Human-readable description

### 2. **User Control Options**
- ✅ **Accept All**: Approve the entire plan
- ❌ **Reject All**: Cancel the integration
- 🔍 **Detailed Review**: Review each step individually
- 📋 **Auto-Approve**: Skip approval with `--auto-approve` flag

### 3. **Individual Step Review**
When choosing detailed review, users can:
- **Approve** individual steps
- **Skip** optional steps  
- **View Details** for technical information
- See **risk assessments** for each action

### 4. **Risk Classification**
- 🟢 **LOW**: Read-only operations, navigation
- 🟡 **MEDIUM**: Configuration changes, creating resources
- 🔴 **HIGH**: Deletions, dangerous operations

### 5. **Step Categories**
- 🔵 **OKTA**: Okta configuration steps
- 🟣 **VENDOR**: Vendor system configuration
- 🟢 **VERIFY**: Testing/verification steps
- ⚪ **GENERAL**: Other operations

## Usage

### Interactive Mode (Default)
```bash
konfig integrate https://vendor-docs.com --okta-domain company.okta.com
```
This will show the plan and ask for approval.

### Auto-Approve Mode
```bash
konfig integrate https://vendor-docs.com --okta-domain company.okta.com --auto-approve
```
This skips user approval and executes all steps automatically.

## Example Plan Display

```
╭──────────────────────────── Integration Overview ────────────────────────────╮
│ 🔗 SSO Integration Plan                                                      │
│                                                                              │
│ Vendor: Google Workspace                                                     │
│ Okta Domain: company.okta.com                                                │
│ Integration Type: SAML 2.0 Single Sign-On                                    │
╰──────────────────────────────────────────────────────────────────────────────╯

                               📋 Integration Plan Summary                      
┏━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━
┃ #   ┃ Step Name                 ┃  Category  ┃   Risk   ┃ Required ┃ Description
┡━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━
│ 1   │ Create Okta Application   │    OKTA    │   MED    │    ✓     │ Create new
│ 2   │ Configure SAML Settings   │   VENDOR   │   MED    │    ✓     │ Configure 
│ 3   │ Navigate to Admin Console │   VENDOR   │   LOW    │    ✓     │ Navigate to
│ 4   │ Test SSO Connection       │   VERIFY   │   LOW    │    ○     │ Test the
└─────┴───────────────────────────┴────────────┴──────────┴──────────┴──────────
```

## Benefits

1. **👁️ Full Visibility**: Users see exactly what the system will do
2. **🎛️ Fine-Grained Control**: Approve/reject individual steps
3. **⚡ Speed Options**: Auto-approve for trusted scenarios  
4. **🛡️ Risk Awareness**: Clear risk levels for each action
5. **📝 Clear Descriptions**: Human-readable explanations
6. **🚫 Safety**: Easy cancellation at any point

## Technical Implementation

### Files Added/Modified

1. **`konfig/services/plan_approval.py`** (NEW)
   - `PlanApprovalService` class
   - Rich console UI for plan display
   - Individual step review logic

2. **`konfig/orchestrator/mvp_orchestrator.py`** (MODIFIED)
   - Updated `integrate_application()` to accept `auto_approve` parameter
   - Updated `_execute_plan()` to use plan approval
   - Auto-approve bypass logic

3. **`konfig/cli.py`** (MODIFIED)
   - Added `--auto-approve` command line flag
   - Pass flag to orchestrator

### Data Structures

```python
@dataclass
class PlanStep:
    id: str
    name: str
    description: str
    tool: str
    action: str
    params: Dict[str, Any]
    category: str  # 'okta', 'vendor', 'verification', 'general'
    risk_level: str  # 'low', 'medium', 'high'
    approved: bool = False
    skipped: bool = False
    required: bool = True
```

## Security Features

- 🔒 **Parameter Sanitization**: Hides passwords/secrets in display
- 🛡️ **Required Step Protection**: Required steps cannot be skipped
- ⚠️ **Risk Warnings**: Clear warnings for high-risk operations
- 📋 **Audit Trail**: All approvals logged

## Future Enhancements

- Save/load approval templates
- Integration-specific approval policies
- Team approval workflows
- Approval history and audit logs