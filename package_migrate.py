"""
迁移llama_index的包
需要先把llama_index代码拉取下来
git clone https://github.com/run-llama/llama_index.git
git clone https://github.com/run-llama/workflows-py.git
"""
import os
import ast
import shutil
from pathlib import Path

PACKAGE_ROOT = "llama_index"
PARENT_PROJECT_PATH = Path(__file__).parent.parent.absolute()
os.makedirs(PACKAGE_ROOT, exist_ok=True)


def extract_py_imports(py_filepath: str):
    """
    提取Python文件中的导入语句
    Args:
        py_filepath (str): Python文件路径
    Returns:
        list: 导入语句列表
    Example:
        imports = extract_py_imports('example.py')
    """
    if not os.path.isfile(py_filepath):
        return []
    print(f"Extracting imports from {py_filepath}")
    # 读取文件内容
    with open(py_filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 解析Python源代码为抽象语法树(AST)
    tree = ast.parse(content)
    imports = []

    # 遍历AST节点，提取所有import语句
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            # 处理 import xxx 形式的语句
            for alias in node.names:
                imports.append({
                    'type': 'import',
                    'module': alias.name,
                    'alias': alias.asname,
                    'lineno': node.lineno
                })
        elif isinstance(node, ast.ImportFrom):
            # 处理 from xxx import yyy 形式的语句
            if node.module and node.level == 0:  # 排除相对导入(带.的导入)
                imports.append({
                    'type': 'from',
                    'module': node.module,
                    'level': node.level,
                    'names': [alias.name for alias in node.names],
                    'aliases': [alias.asname for alias in node.names],
                    'lineno': node.lineno
                })
    return imports


def update_py_file_import(py_filepath: str, mapping: dict[str, str]):
    """
    更新Python文件中的导入语句，根据提供的映射规则替换模块路径
    Args:
        py_filepath (str): Python文件路径
        mapping (Dict[str, str]): 模块路径映射字典，格式为 {旧模块路径: 新模块路径}
    Returns:
        list: 发生变更的导入语句列表，包含变更详情
    Example:
        mapping = {'old.package': 'new.package'}
        changes = update_file_import('example.py', mapping)
    """
    # 检查文件是否存在且为Python文件
    if not os.path.isfile(py_filepath) or not py_filepath.endswith('.py'):
        print(f"{py_filepath} is not a valid Python file")
        return []

    imports = extract_py_imports(py_filepath)
    changes = []
    # 重新读取文件行内容，准备修改
    with open(py_filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 按行号倒序处理，避免修改前面行影响后面行的行号
    for imp in sorted(imports, key=lambda x: x['lineno'], reverse=True):
        lineno = imp['lineno'] - 1  # 转换为0索引
        old_module = imp['module']

        # 查找是否有匹配的映射规则
        new_module = None
        for key in mapping:
            # 检查是否完全匹配或者是否为子模块（确保是完整的模块名前缀）
            if old_module == key or old_module.startswith(key + '.'):
                new_module = mapping[key] + old_module[len(key):]
                break
        # 如果没有匹配的映射规则，则跳过
        if new_module is None:
            continue

        if imp['type'] == 'import':
            # 处理 import xxx 语句, 替换整行中的模块名
            new_line = lines[lineno].replace(f"import {old_module}", f"import {new_module}")
            lines[lineno] = new_line
            changes.append(f"Line {imp['lineno']}: 'import {old_module}' -> 'import {new_module}'")
        elif imp['type'] == 'from' and imp['level'] == 0:
            # 处理 from xxx import yyy 语句, 替换from子句中的模块名
            new_line = lines[lineno].replace(f"from {old_module}", f"from {new_module}")
            lines[lineno] = new_line
            changes.append(f"Line {imp['lineno']}: 'from {old_module}' -> 'from {new_module}'")

    # 如果有变更，则写回文件
    if changes:
        with open(py_filepath, 'w', encoding='utf-8') as f:
            f.writelines(lines)
    for change in changes:
        print(change)
    return changes


def migrate_package_files(
        source_package_dir: str, target_package_dir: str,
        mapping: dict[str, str],
        ignore_files: list[str] = None,
        clear_target=True,
):
    """
    迁移包文件，将源目录中的所有文件复制到目标目录，并更新其中的导入语句

    Args:
        source_package_dir (str): 源包目录路径
        target_package_dir (str): 目标包目录路径
        mapping (dict[str, str]): 模块路径映射字典，格式为 {旧模块路径: 新模块路径}
        ignore_files (list[str]): 忽略迁移的文件列表
        clear_target (bool): 是否清空目标目录，默认为 True
    Returns:
        bool: 迁移成功返回True，失败返回False

    Example:
        mapping = {'old.package': 'new.package'}
        migrate_package_files('./source_pkg', './target_pkg', mapping)
    """
    # 检查源目录是否存在
    if not os.path.isdir(source_package_dir):
        print(f"Error: {source_package_dir} is not a valid directory")
        return False

    # 如果目标目录已存在，先删除再重新创建
    if os.path.isdir(target_package_dir) and clear_target:
        print(f"{target_package_dir} already exists, clearing it...")
        shutil.rmtree(target_package_dir)

    # 创建目标目录
    os.makedirs(target_package_dir, exist_ok=True)

    # 转换为Path对象便于操作
    source_package_path = Path(source_package_dir)

    # 遍历源目录中的所有文件
    for file in source_package_path.rglob('*'):
        # 只处理文件，跳过目录
        if not file.is_file():
            continue
        # 检查是否在忽略列表中
        if ignore_files:
            # 获取相对于源目录的路径
            relative_file_path = file.relative_to(source_package_path).as_posix()
            # 检查文件路径是否以忽略列表中的任意项开头
            if any(relative_file_path.startswith(ignore_path) for ignore_path in ignore_files):
                print(f"Ignoring {file}")
                continue
        # 计算目标文件路径（保持相对目录结构）
        relative_path = file.relative_to(source_package_path)
        target_file = Path(target_package_dir) / relative_path

        # 确保目标文件的目录存在
        target_file.parent.mkdir(parents=True, exist_ok=True)

        # 复制文件
        shutil.copy2(file, target_file)  # 使用copy2保留文件元数据
        print(f"Copied {file} to {target_file}")

        # 如果是Python文件，则更新其中的导入语句
        if str(file).endswith('.py'):
            try:
                update_py_file_import(str(target_file), mapping)
            except Exception as e:
                print(f"Warning: Error updating imports in {file}: {e}")

    print(f"Package migration completed: {source_package_dir} -> {target_package_dir}")
    return True


def migrate_instrumentation():
    source_package_dir = os.path.join(
        PARENT_PROJECT_PATH,
        'llama_index',
        r'llama-index-instrumentation\src\llama_index_instrumentation'
    )
    target_package_dir = r'llama_index\core\instrumentation'
    mapping = {
        'llama_index_instrumentation': 'llama_index.core.instrumentation'
    }
    ignore_files = [
        'server', 'testing'
    ]
    migrate_package_files(source_package_dir, target_package_dir, mapping, ignore_files)


def migrate_workflow():
    source_package_dir = os.path.join(
        PARENT_PROJECT_PATH,
        'workflow_py',
        r'src\workflows'
    )
    target_package_dir = r'llama_index\core\instrumentation'
    mapping = {
        'workflows': 'llama_index.core.workflow'
    }
    migrate_package_files(source_package_dir, target_package_dir, mapping)


if __name__ == '__main__':
    migrate_instrumentation()
