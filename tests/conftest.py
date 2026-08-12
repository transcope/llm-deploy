import os
import sys

# 将 llm_deploy 目录加入模块搜索路径，方便测试导入
_SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "llm_deploy")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
