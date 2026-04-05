What stays in core vs what becomes a package:
tau/core/           ← KEEP SMALL (already is)
├── agent.py        ← Agent loop (stays)
├── context.py      ← Context management (stays)
├── session.py      ← Session persistence (stays)
├── extension.py    ← Extension system (stays)
├── tool_registry.py← Tool dispatch (stays)
└── types.py        ← Shared types (stays)
tau/tools/          ← KEEP MINIMAL (just fs + shell)
├── fs.py           ← read/write/edit/grep/glob
└── shell.py        ← bash execution

Everything else becomes an extension package:
tau-agents/              ← git package: tau install git:github.com/datctbk/tau-agents
├── tau.json
├── extensions/
│   └── agent_tool/
│       └── extension.py   ← AgentTool, SendMessage, TaskCreate/Stop
├── skills/
│   └── built-in-agents/
│       ├── explore.md     ← Read-only research agent
│       ├── plan.md        ← Planning agent
│       └── verify.md      ← Verification agent
tau-memory/              ← git package: tau install git:github.com/datctbk/tau-memory
├── extensions/
│   └── memory/
│       └── extension.py   ← Injects MEMORY.md into system prompt, autoDream
tau-web/                 ← git package: tau install git:github.com/datctbk/tau-web
├── extensions/
│   └── web/
│       └── extension.py   ← WebFetchTool, WebSearchTool
tau-mcp/                 ← git package
├── extensions/
│   └── mcp/
│       └── extension.py   ← MCP client, MCPTool, ListResources