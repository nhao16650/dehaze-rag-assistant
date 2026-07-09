from pathlib import Path

# 项目根目录：dehaze-rag-assistant/
BASE_DIR = Path(__file__).resolve().parents[2]

# 数据目录
PAPERS_DIR = BASE_DIR / "papers"
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "index"
LOG_DIR = DATA_DIR / "logs"

# 向量库文件
FAISS_INDEX_PATH = INDEX_DIR / "dehaze_faiss.index"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
CHUNKS_PATH = INDEX_DIR / "chunks.json"

# Embedding 模型
# 英文论文优先用这个，小、快、容易跑通
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# 文本切分参数
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# 检索返回片段数量
TOP_K = 5