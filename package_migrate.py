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
        import_mapping={},
        str_mapping={
            'from llama_index_instrumentation import': 'from .. import',
            'from llama_index_instrumentation.': 'from ..',
        },
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


def migrate_sub_packages():
    # 定义需要迁移的包目录
    source_packages = [
        # LLMs packages
        r'llama-index-integrations\llms\llama-index-llms-openai\llama_index\llms\openai',
        r'llama-index-integrations\llms\llama-index-llms-openai-like\llama_index\llms\openai_like',
        r'llama-index-integrations\llms\llama-index-llms-dashscope\llama_index\llms\dashscope',
        # Tools packages
        r'llama-index-integrations\tools\llama-index-tools-mcp\llama_index\tools\mcp',
        # Utils packages
        r'llama-index-utils\llama-index-utils-workflow\llama_index\utils\workflow',
        # Memory packages
        r'llama-index-integrations\memory\llama-index-memory-mem0\llama_index\memory\mem0',
    ]

    for source_dir in source_packages:
        # 从路径中提取目标目录名称
        # 例如从 r'llama-index-integrations\llms\llama-index-llms-openai\llama_index\llms\openai'
        # 提取 'llms' 和 'openai'
        path_parts = source_dir.split('\\')
        target_parent_dir = path_parts[-2]  # 倒数第二个是父目录名 (如 'llms', 'tools' 等)
        target_child_dir = path_parts[-1]   # 最后一个是子目录名 (如 'openai', 'mcp' 等)

        migrate_package_files(
            source_package_dir=os.path.join(PARENT_PROJECT_PATH, 'llama_index', source_dir),
            import_mapping=MIGRATE_IMPORT_MAPPING,
            target_package_dir=os.path.join('llama_index', target_parent_dir, target_child_dir),
            clear_target=True,
        )


def migrate_core(migrate_switch: bool = False):
    # 排除掉废弃和没什么用的模块
    exclude_modules = [
        # 已经迁移的基础模块
        'workflow', 'instrumentation',
        # 废弃的模块
        'command_line', 'text_splitter',
        # 不用的模块
        'composability', 'sparse_embeddings', 'service_context_elements',
        'playground', 'chat_ui', 'voice_agents', 'llama_pack',
        'langchain_helpers/text_splitter.py',
    ]
    if migrate_switch:
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

    # 检查排除掉的模块是否存在依赖关系
    base_modules = ['instrumentation', 'workflow']
    dependency_map = extract_package_dependencies(
        os.path.join(PACKAGE_ROOT, 'core'),
        include_prefixes=[f'llama_index.core.{x}' for x in exclude_modules
                          if x not in base_modules and not x.endswith('.py')],
        exclude_prefixes=[f'llama_index.core.{x}' for x in base_modules],
    )
    for package, py_files in sorted(dependency_map.items(), key=lambda item: len(item[1])):
        print(f'Depend: {package}, file_count: {len(py_files)}, files: {py_files}')


def migrate_core_simple(migrate_switch: bool = False):
    # 只迁移核心模块
    include_modules = [
        'agent', 'llms', 'memory', 'prompts', 'base', 'bridge', 'storage', 'tools', 'schema',
        '__init__.py', 'constants.py', 'py.typed', 'settings.py', 'types.py', 'utils.py', 'async_utils.py',
    ]
    if migrate_switch:
        migrate_package_files(
            source_package_dir=os.path.join(
                PARENT_PROJECT_PATH,
                'llama_index',
                r'llama-index-core\llama_index\core'
            ),
            target_package_dir=r'llama_index\core',
            import_mapping=MIGRATE_IMPORT_MAPPING,
            include_files=include_modules,
        )

    # 检查迁移后的是否还依赖剩余的模块
    check_modules = ['instrumentation', 'workflow']
    check_modules.extend(include_modules)
    dependency_map = extract_package_dependencies(
        os.path.join(PACKAGE_ROOT, 'core'),
        include_prefixes=['llama_index'],
        exclude_prefixes=[f'llama_index.core.{x}' for x in check_modules if not x.endswith('.py')],
    )
    py_file_depend_map = {}
    for package, py_files in sorted(dependency_map.items(), key=lambda item: len(item[1])):
        print(f'Depend: {package}, file_count: {len(py_files)}, files: {py_files}')
        for py_file in py_files:
            if py_file not in py_file_depend_map:
                py_file_depend_map[py_file] = []
            py_file_depend_map[py_file].append(package)
    for py_file, depend_modules in sorted(py_file_depend_map.items(), key=lambda item: len(item[1])):
        print(f'File: {py_file} depend count: {len(depend_modules)}: {depend_modules}')


def migrate_tests():
    migrate_package_files(
        source_package_dir=os.path.join(
            PARENT_PROJECT_PATH,
            'llama_index',
            r'llama-index-core\tests\agent'
        ),
        target_package_dir=r'tests\agent',
        import_mapping=MIGRATE_IMPORT_MAPPING,
        clear_target=True,
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
    shutil.rmtree(os.path.join(PACKAGE_ROOT, 'core'), ignore_errors=True)
    migrate_instrumentation()
    migrate_workflow()
    migrate_sub_packages()
    migrate_tests()
    migrate_core(True)
    # migrate_core_simple(True)
