"""语义记忆实现

结合向量检索和知识图谱的混合语义记忆，使用：
- HuggingFace 中文预训练模型进行文本嵌入
- 向量相似度检索进行快速初筛
- 知识图谱进行实体关系推理
- 混合检索策略优化结果质量
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
import json
import logging
import math
import numpy as np

from ..base import BaseMemory, MemoryItem, MemoryConfig
from ..embedding import get_text_embedder, get_dimension


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Entity:
    """实体类"""
    def __init__(
            self,
            entity_id: int,
            name: str,
            entity_type: str,
            description: str,
            properties: Dict[str, Any] = None
    ):
        self.entity_id = entity_id
        self.name = name
        self.entity_type = entity_type  # PERSON, ORG, PRODUCT, SKILL, CONCEPT等
        self.description = description
        self.properties = properties or {}
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.frequency = 1      # 出现频率

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "entity_type": self.entity_type,
            "description": self.description,
            "properties": self.properties,
            "frequency": self.frequency
        }

class Relation:
    """关系类"""
    def __init__(
            self,
            from_entity: str,
            to_entity: str,
            relation_type: str,
            strength: float = 1.0,
            evidence: str = "",
            properties: Dict[str, Any] = None
    ):
        self.from_entity = from_entity
        self.to_entity = to_entity
        self.relation_type = relation_type
        self.strength = strength
        self.evidence = evidence    # 支持该关系的原文本
        self.properties = properties or {}
        self.created_at = datetime.now()
        self.frequency = 1  # 关系出现频率

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_entity": self.from_entity,
            "to_entity": self.to_entity,
            "relation_type": self.relation_type,
            "strength": self.strength,
            "evidence": self.evidence,
            "properties": self.properties,
            "frequency": self.frequency
        }

class SemanticMemory(BaseMemory):
    """
    增强语义记忆实现

    特点：
    - 使用HuggingFace中文预训练模型进行文本嵌入
    - 向量检索进行快速相似度匹配
    - 知识图谱存储实体和关系
    - 混合检索策略：向量+图+语义推理
    """
    def __init__(self, config: MemoryConfig, storage_backend=None):
        super().__init__(config, storage_backend)

        # 嵌入模型（统一提供）
        self.embedding_model = None
        self._init_embedding_model()

        # 专业数据库存储
        self.vector_store = None
        self.graph_store = None
        self._init_databases()

        # 实体和关系存储（用于快速访问）
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []

        # 实体识别器
        self.nlp = None
        self._init_nlp()

        # 记忆存储
        self.semantic_memories: List[MemoryItem] = []
        self.memory_embeddings: Dict[str, np.ndarray] = {}

        logger.info("增强语义记忆初始化完成（使用Qdrant+Neo4j专业数据库）")


    def _init_embedding_model(self):
        """初始化统一嵌入模型（由 embedding_provider 管理）。"""
        try:
            self.embedding_model = get_text_embedder()
            # 轻量健康检查与日志
            try:
                test_vec = self.embedding_model.encode("health_check")
                dim = getattr(self.embedding_model, "dimension", len(test_vec))
                logger.info(f"✅ 嵌入模型就绪，维度: {dim}")
            except Exception:
                logger.info("✅ 嵌入模型就绪")
        except Exception as e:
            logger.error(f"❌ 嵌入模型初始化失败: {e}")
            raise

    def _init_databases(self):
        """初始化专业数据库存储"""
        try:
            from ...core.database_config import get_database_config
            # 获取数据库配置
            db_config = get_database_config()

            # 初始化Qdrant向量数据库（使用连接管理器避免重复连接）
            from ..storage.qdrant_store import QdrantConnectionManager
            qdrant_config = db_config.get_qdrant_config() or {}
            qdrant_config["vector_size"] = get_dimension()
            self.vector_store = QdrantConnectionManager.get_instance(**qdrant_config)
            logger.info("✅ Qdrant向量数据库初始化完成")

            # 初始化Neo4j图数据库
            from ..storage.neo4j_store import Neo4jGraphStore
            neo4j_config = db_config.get_neo4j_config()
            self.graph_store = Neo4jGraphStore(**neo4j_config)
            logger.info("✅ Neo4j图数据库初始化完成")

            # 验证连接
            vector_health = self.vector_store.health_check()
            graph_health = self.graph_store.health_check()

            if not vector_health:
                logger.warning("⚠️ Qdrant连接异常，部分功能可能受限")
            if not graph_health:
                logger.warning("⚠️ Neo4j连接异常，图搜索功能可能受限")

            logger.info(
                f"🏥 数据库健康状态: Qdrant={'✅' if vector_health else '❌'}, Neo4j={'✅' if graph_health else '❌'}")

        except Exception as e:
            logger.error(f"❌ 数据库初始化失败: {e}")
            logger.info("💡 请检查数据库配置和网络连接")
            logger.info("💡 参考 DATABASE_SETUP_GUIDE.md 进行配置")
            raise

    def _init_nlp(self):
        """初始化NLP处理器 - 智能多语言支持"""
        try:
            import spacy
            self.nlp_models = {}

            # 尝试加载多语言模型
            models_to_try = [
                ("zh_core_web_sm", "中文"),
                ("en_core_web_sm", "英文")
            ]

            loaded_models = []
            for model_name, lang_name in models_to_try:
                try:
                    nlp = spacy.load(model_name)
                    self.nlp_models[model_name] = nlp
                    loaded_models.append(lang_name)
                    logger.info(f"✅ 加载{lang_name}spaCy模型: {model_name}")
                except OSError:
                    logger.warning(f"⚠️ {lang_name}spaCy模型不可用: {model_name}")

            # 设置主要NLP处理器
            if "zh_core_web_sm" in self.nlp_models:
                self.nlp = self.nlp_models["zh_core_web_sm"]
                logger.info("🎯 主要使用中文spaCy模型")
            elif "en_core_web_sm" in self.nlp_models:
                self.nlp = self.nlp_models["en_core_web_sm"]
                logger.info("🎯 主要使用英文spaCy模型")
            else:
                self.nlp = None
                logger.warning("⚠️ 无可用spaCy模型，实体提取将受限")

            if loaded_models:
                logger.info(f"📚 可用语言模型: {', '.join(loaded_models)}")

        except ImportError:
            logger.warning("⚠️ spaCy不可用，实体提取将受限")
            self.nlp = None
            self.nlp_models = {}

    def add(self, memory_item: MemoryItem) -> str:
        """添加语义记忆"""
        try:
            # 1. 生成文本嵌入
            embedding = self.embedding_model.encode(memory_item.content)
            self.memory_embeddings[memory_item.id] = embedding

            # todo: 2. 提取实体和关系

            # 3. 存储到Neo4j数据库

            # 4. 存储到Qdrant数据库
        except Exception as e:
            logger.error(f"❌ 添加语义记忆失败: {e}")
            raise
