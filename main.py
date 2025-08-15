import json
import time
import random
import pydantic
from astrbot.api.event import filter
from astrbot.api.star import Context, Star, register
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.api import logger
from .sentiment import Sentiment
from .similarity import Similarity


class MemberState(pydantic.BaseModel):
    uid: str
    last_wake: int = 0  # 最后唤醒bot的时间
    silence_until: int = 0  # 被辱骂后沉默到何时


class GroupState(pydantic.BaseModel):
    gid: str
    members: dict[str, MemberState] = {}  # uin -> state
    shutup_until: int = 0  # 闭嘴到何时


class StateManager:
    """内存状态管理"""

    _groups: dict[str, GroupState] = {}

    @classmethod
    def get_group(cls, gid: str) -> GroupState:
        if gid not in cls._groups:
            cls._groups[gid] = GroupState(gid=gid)
        return cls._groups[gid]

    @staticmethod
    def now() -> int:
        return int(time.time())


@register(
    "astrbot_plugin_wakepro",
    "Zhalslar",
    "更强大的唤醒增强插件：提及唤醒、唤醒延长、话题相关性唤醒、答疑唤醒、无聊唤醒、闭嘴机制、被骂沉默机制",
    "v1.0.0",
)
class WakeProPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.conf = config
        self.sent = Sentiment()

    def _is_shutup(self, g: GroupState) -> bool:
        return g.shutup_until > StateManager.now()

    def _is_insult(self, g: GroupState, uin: str) -> bool:
        m = g.members.get(uin)
        silence_until = m.silence_until if m else 0
        return silence_until > StateManager.now()

    async def _get_history_msg(
        self, event: AstrMessageEvent, role: str = "assistant", count: int = 10
    ) -> list | None:
        """获取历史消息"""
        try:
            umo = event.unified_msg_origin
            curr_cid = await self.context.conversation_manager.get_curr_conversation_id(
                umo
            )
            conversation = await self.context.conversation_manager.get_conversation(
                umo, curr_cid
            )
            history = json.loads(conversation.history)
            contexts = []
            for record in history:
                if record["role"] == role:
                    if "content" in record and record["content"]:
                        contexts.append({record["content"]})

            contexts = [item for sublist in contexts for item in sublist]
            return contexts[-count:]

        except Exception as e:
            logger.error(f"获取历史消息失败：{e}")
            return None

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def on_group_msg(self, event: AstrMessageEvent):
        """主入口"""
        bid: str = event.get_self_id()
        gid: str = event.get_group_id()
        uid: str = event.get_sender_id()
        msg: str = event.message_str.strip()
        g: GroupState = StateManager.get_group(gid)

        # 1. 自身消息 / 群聊白名单 / 用户黑名单
        if uid == bid:
            event.stop_event()
            return
        if gid and self.conf["group_whitelist"] and gid not in self.conf["group_whitelist"]:
            return
        if uid in self.conf.get("user_blacklist", []):
            return

        # 2. 更新成员状态
        if uid not in g.members:
            g.members[uid] = MemberState(uid=uid)

        # 3. 沉默 / 闭嘴检查
        if self._is_shutup(g):
            event.stop_event()
            return
        if self._is_insult(g, uid):
            event.stop_event()
            return

        # 4. 各类唤醒条件
        should_wake = False
        reason = None

        # 4.1 提及唤醒
        if self.conf["mention_wake"]:
            names = [n for n in self.conf["mention_wake"] if n]
            for n in names:
                if n and n in msg:
                    should_wake = True
                    reason = f"提及唤醒({n})"
                    break

        # 4.2 唤醒延长（如果已经处于唤醒状态且在 wake_extend 秒内，每个用户单独延长唤醒时间）
        if (
            not should_wake
            and self.conf["wake_extend"]
            and (StateManager.now() - g.members[uid].last_wake)
            <= int(self.conf["wake_extend"] or 0)
        ):
            should_wake = True
            reason = "唤醒延长"

        # 4.3 话题相关性唤醒
        if not should_wake and self.conf["relevant_wake"] :
            if bmsgs := await self._get_history_msg(event, count=5):
                for bmsg in bmsgs:
                    simi = Similarity.cosine(msg, bmsg, gid)
                    if simi > self.conf["relevant_wake"]:
                        should_wake = True
                        reason = f"话题相关性{simi}>{self.conf['relevant_wake']}"
                        break

        # 4.4 答疑唤醒
        if not should_wake and self.conf["ask_wake"]:
            if self.sent.ask(msg) > self.conf["ask_wake"]:
                should_wake = True
                reason = "答疑唤醒"

        # 4.5 无聊唤醒
        if not should_wake and self.conf["bored_wake"]:
            if self.sent.bored(msg) > self.conf["bored_wake"]:
                should_wake = True
                reason = "无聊唤醒"

        # 4.6 概率唤醒
        if not should_wake and self.conf["prob_wake"]:
            if random.random() < self.conf["prob_wake"]:
                should_wake = True
                reason = "概率唤醒"

        # 5. 触发唤醒
        if should_wake:
            event.is_at_or_wake_command = True
            g.members[uid].last_wake = StateManager.now()
            logger.info(f"[wakepro] 群({gid}){reason}：{msg}")

        # 6. 闭嘴机制(对当前群聊闭嘴)
        if self.conf["shutup"]:
            shut_th = self.sent.shut(msg)
            if shut_th > self.conf["shutup"]:
                shut_sec = int(shut_th * self.conf["sult_multiple"])
                g.shutup_until = StateManager.now() + shut_sec
                reason = f"触发闭嘴机制{shut_sec}秒"
                logger.info(f"[wakepro] 群({gid}){reason}：{msg}")
                return

        # 7. 沉默机制(对单个用户沉默)
        if self.conf["insult"]:
            insult_th = self.sent.insult(msg)
            if insult_th > self.conf["insult"]:
                silence_sec = int(insult_th * self.conf["sult_multiple"])
                g.members[uid].silence_until = StateManager.now() + silence_sec
                reason = f"触发沉默机制{silence_sec}秒"
                logger.info(f"[wakepro] 群({gid})用户({uid}){reason}：{msg}")
                return
