# shared.py
import asyncio
from contextlib import AsyncExitStack
from typing import TypedDict, Annotated, List, Optional
import operator

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPSessionPool:
    """
    Tek bir stdio MCP session'ı, eşzamanlı gelen tüm /chat isteklerini
    (ve arkaplan push-bildirim görevlerini) sıraya sokan bir darboğazdı.

    Bu sınıf, aynı server.py'yi ayrı alt süreçler olarak `pool_size` kez
    başlatıp her biri için bağımsız bir ClientSession açar. call_tool() bir
    asyncio.Queue üzerinden o an boşta olan bir session'ı alır, çağrıyı yapar
    ve session'ı havuza geri bırakır — böylece paralel istekler birbirini
    beklemek zorunda kalmaz. Havuzdaki tüm session'lar meşgulse bir sonraki
    çağrı otomatik olarak sırada bekler (backpressure), sistemi olduğu gibi
    tıkamak yerine.

    Arayüzü tek bir mcp.ClientSession ile birebir aynıdır
    (`await pool.call_tool(name, args)`), bu yüzden client.py ve api.py'de
    `ctx.session`'ı kullanan hiçbir yer değişmek zorunda kalmaz.
    """

    def __init__(self, pool_size: int = 3):
        self.pool_size = pool_size
        self._exit_stack = AsyncExitStack()
        self._sessions: list[ClientSession] = []
        self._available: "asyncio.Queue[ClientSession]" = asyncio.Queue()
        self.tools: list[dict] = []

    async def start(self, server_params: StdioServerParameters, tool_loader):
        """
        server_params: alt süreç olarak başlatılacak MCP server komutu.
        tool_loader  : async def(session) -> list[dict] (client.py::get_ollama_tools).
        """
        for _ in range(self.pool_size):
            read, write = await self._exit_stack.enter_async_context(stdio_client(server_params))
            session = await self._exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self._sessions.append(session)
            self._available.put_nowait(session)

        # Tüm session'lar aynı tool setini sunar (aynı server.py); ilkinden yükle.
        self.tools = await tool_loader(self._sessions[0])

    async def call_tool(self, name: str, args: dict):
        session = await self._available.get()
        try:
            return await session.call_tool(name, args)
        finally:
            self._available.put_nowait(session)

    def __bool__(self):
        # `if not ctx.session:` kontrolünün, session'lar hazır olana kadar
        # False dönmesi için: henüz hiç session eklenmediyse pool "boş" sayılır.
        return len(self._sessions) > 0

    async def aclose(self):
        await self._exit_stack.aclose()


class AppContext:
    def __init__(self):
        self.session = None          # MCPSessionPool örneği (lifespan içinde atanır)
        self.tools = None
        self.db_pool = None           # asyncpg.Pool (lifespan içinde atanır)
        self.exit_stack = AsyncExitStack()

# Ortak context objesi
ctx = AppContext()

# Ortak durum takibi için global sözlük
chat_statuses = {}

# Ortak tip tanımı
class GraphState(TypedDict):
    prompt: str
    persona: str
    intent: str
    messages: Annotated[List[dict], operator.add]
    final_output: str
    tools: List[dict]
    tool_calls: Optional[List[dict]]   # Hangi tool'lar çağrıldı, hangi argümanlarla
    tool_results: Optional[List[dict]] # MCP server'dan dönen ham veriler
    session_id: Optional[str]          # İstek oturum ID'si
    intent_data: Optional[dict]