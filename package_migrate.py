"""
迁移llama_index的包
需要先把llama_index代码拉取下来
git clone https://github.com/run-llama/llama_index.git
git clone https://github.com/run-llama/workflows-py.git
"""
import os
import shutil
from pathlib import Path

from migrate_util import migrate_package_files, extract_package_dependencies

PACKAGE_ROOT = "llama_index"
PARENT_PROJECT_PATH = Path(__file__).parent.parent.absolute()
os.makedirs(PACKAGE_ROOT, exist_ok=True)
MIGRATE_IMPORT_MAPPING = {
    'llama_index_instrumentation': 'llama_index.core.instrumentation',
    # 没有 events/base.py ，修改包引用
    'llama_index.core.instrumentation.events.base': 'llama_index.core.instrumentation.base',
}


def migrate_instrumentation():
    # 先将core目录下的一些event定义进行迁移
    target_package_dir = r'llama_index\core\instrumentation'
    migrate_package_files(
        source_package_dir=os.path.join(
            PARENT_PROJECT_PATH, 'llama_index',
            r'llama-index-core\llama_index\core\instrumentation'
        ),
        target_package_dir=target_package_dir,
        import_mapping=MIGRATE_IMPORT_MAPPING,
        clear_target=True,
        ignore_pure_file=True,
    )

    # 将 llama-index-instrumentation 子模块的部分迁移过去
    migrate_package_files(
        source_package_dir=os.path.join(
            PARENT_PROJECT_PATH, 'llama_index',
            r'llama-index-instrumentation\src\llama_index_instrumentation'
        ),
        target_package_dir=target_package_dir,
        exclude_files=['events/__init__.py'],
        import_mapping=MIGRATE_IMPORT_MAPPING,
    )


def migrate_workflow():
    migrate_package_files(
        source_package_dir=os.path.join(
            PARENT_PROJECT_PATH, 'llama_index',
            r'llama-index-core\llama_index\core\workflow'
        ),
        target_package_dir=r'llama_index\core\workflow',
        import_mapping=MIGRATE_IMPORT_MAPPING,
        exclude_files=[
            'drawing.py',   # 工作流绘图工具，依赖pyvis，已废弃
            'context_serializers.py',   # 兼容旧版的，不需要
        ],
        str_mapping={
            'from .context_serializers import JsonPickleSerializer, JsonSerializer':
                'from .context import JsonSerializer, PickleSerializer as JsonPickleSerializer',
        },
        clear_target=True,
        ignore_pure_file=True,
    )
    migrate_package_files(
        source_package_dir=os.path.join(PARENT_PROJECT_PATH, 'workflows-py', r'src\workflows'),
        target_package_dir=r'llama_index\core\workflow',
        import_mapping=MIGRATE_IMPORT_MAPPING,
        str_mapping={
            'from workflows import': 'from .. import',
            'from workflows.': 'from ..',
        },
        # __init__.py 不覆盖，server和testing用不到不迁移
        exclude_files=['server', 'testing', '__init__.py']
    )


def migrate_llm():
    llm_dirs = [
        r'llama-index-integrations\llms\llama-index-llms-openai\llama_index\llms\openai',
        r'llama-index-integrations\llms\llama-index-llms-openai-like\llama_index\llms\openai_like',
        r'llama-index-integrations\llms\llama-index-llms-dashscope\llama_index\llms\dashscope',
    ]
    for llm_dir in llm_dirs:
        migrate_package_files(
            source_package_dir=os.path.join(PARENT_PROJECT_PATH, 'llama_index', llm_dir),
            import_mapping=MIGRATE_IMPORT_MAPPING,
            target_package_dir=r'llama_index\llms\\' + llm_dir.split('\\')[-1],
            clear_target=True,
        )


def migrate_core(exclude_modules: list[str] = None):
    exclude_modules = exclude_modules or []
    exclude_modules.extend(['instrumentation', 'workflow'])
    exclude_modules.extend([
        'readers/download.py',
        'tools/download.py',
        'langchain_helpers/text_splitter.py',
    ])
    migrate_package_files(
        source_package_dir=os.path.join(
            PARENT_PROJECT_PATH,
            'llama_index',
            r'llama-index-core\llama_index\core'
        ),
        target_package_dir=r'llama_index\core',
        import_mapping=MIGRATE_IMPORT_MAPPING,
        exclude_files=exclude_modules,
        clear_target=False,
    )


def check_import_dependencies(check_modules: list[str] = None):
    """检查依赖关系"""
    # 使用 mypy 命令也可以检查import问题
    check_modules = check_modules or []
    dependency_map = extract_package_dependencies(
        os.path.join(PACKAGE_ROOT, 'core'),
        include_prefixes=[f'llama_index.core.{x}' for x in check_modules],
        exclude_prefixes=['llama_index.core.instrumentation', 'llama_index.core.workflow'],
    )
    for package, py_files in dependency_map.items():
        print(f'Depend: {package}, file_count: {len(py_files)}, files: {py_files}')


if __name__ == '__main__':
    # shutil.rmtree(os.path.join(PACKAGE_ROOT, 'core'), ignore_errors=True)
    # migrate_instrumentation()
    # migrate_workflow()
    # 不需要的模块
    core_exclude_modules = [
        # 废弃的模块
        'command_line', 'text_splitter',
        # 不用的模块
        'composability', 'sparse_embeddings', 'service_context_elements',
        'playground', 'chat_ui', 'voice_agents', 'llama_pack'
    ]
    # migrate_core(core_exclude_modules)
    check_import_dependencies(core_exclude_modules)
    # migrate_llm()
