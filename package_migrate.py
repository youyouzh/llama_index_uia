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
    'llama_index_instrumentation': 'llama_index.core.instrumentation'
}


def migrate_instrumentation():
    # 先将core目录下的一些event定义进行迁移
    migrate_package_files(
        source_package_dir=os.path.join(
            PARENT_PROJECT_PATH,
            'llama_index',
            r'llama-index-core\llama_index\core\instrumentation'
        ),
        target_package_dir=r'llama_index\core\instrumentation',
        import_mapping={},
        clear_target=True,
        ignore_pure_file=True,
    )

    # 将 llama-index-instrumentation 子模块的部分迁移过去
    migrate_package_files(
        source_package_dir=os.path.join(
            PARENT_PROJECT_PATH,
            'llama_index',
            r'llama-index-instrumentation\src\llama_index_instrumentation'
        ),
        target_package_dir=r'llama_index\core\instrumentation',
        import_mapping=MIGRATE_IMPORT_MAPPING,
        clear_target=False,
    )


def migrate_workflow():
    source_package_dir = os.path.join(
        PARENT_PROJECT_PATH,
        'llama_index',
        r'llama-index-core\llama_index\core\workflow'
    )
    migrate_package_files(
        source_package_dir=source_package_dir,
        target_package_dir=r'llama_index\core\workflow',
        import_mapping=MIGRATE_IMPORT_MAPPING,
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
        exclude_files=['server', 'testing'],
        clear_target=False,
    )
    # workflow导出包定义文件保留
    shutil.copy2(
        os.path.join(source_package_dir, '__init__.py'),
        os.path.join(PACKAGE_ROOT, r'core\workflow\__init__.py')
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


def migrate_core_dependencies():
    migrate_package_files(
        source_package_dir=os.path.join(
            PARENT_PROJECT_PATH,
            'llama_index',
            r'llama-index-core\llama_index\core'
        ),
        target_package_dir=r'llama_index\core',
        import_mapping=MIGRATE_IMPORT_MAPPING,
        include_files=[
        ],
        clear_target=False,
    )


def migrate_core_all():
    migrate_package_files(
        source_package_dir=os.path.join(
            PARENT_PROJECT_PATH,
            'llama_index',
            r'llama-index-core\llama_index\core'
        ),
        target_package_dir=r'llama_index\core',
        import_mapping=MIGRATE_IMPORT_MAPPING,
        exclude_files=[
            # 合并的模块
            'instrumentation', 'workflow',
            # 废弃的模块
            'command_line', 'text_splitter',
            # 不用的模块
            'composability', 'download', 'sparse_embeddings', 'service_context_elements',
            'playground', 'langchain_helpers'

        ],
        clear_target=False,
    )


def check_import_dependencies():
    """检查依赖关系"""
    # 使用 mypy 命令也可以检查import问题
    dependencies = extract_package_dependencies(
        os.path.join(PACKAGE_ROOT, 'core\\agent'),
        include_prefixes=['llama_index'],
        exclude_prefixes=[
            'llama_index.core.workflow', 'llama_index.core.agent',
            'llama_index.core.instrumentation', 'llama_index.core.agent',
        ],
    )
    for dependency in dependencies:
        print(dependency)


if __name__ == '__main__':
    # migrate_instrumentation()
    # migrate_workflow()
    # migrate_llm()
    migrate_core_dependencies()
    # check_import_dependencies()
