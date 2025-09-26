"""
包迁移重构工具
"""
import os
import ast
import shutil
from pathlib import Path


def parse_py_file(py_filepath: str) -> dict:
    """
    解析Python文件，返回文件的各种属性
    Args:
        py_filepath (str): Python文件路径
    Returns:
        dict: 包含文件属性的字典
            - imports: 导入语句列表
            - is_pure_import: 是否只包含导入语句
            - is_empty_content: 是否为空文件
    """
    if not os.path.isfile(py_filepath):
        return {
            'imports': [],
            'is_pure_import': False,
            'is_empty_content': False,
        }

    with open(py_filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 解析Python源代码为抽象语法树(AST)
    tree = ast.parse(content)
    imports = []

    # 用于判断是否为纯导入文件的标志
    is_pure_import = True

    # 遍历AST节点，同时提取所有import语句并判断是否为纯import文件
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
        # 同时检查是否包含非导入语句
        elif not isinstance(node, (ast.Module, ast.Import, ast.ImportFrom, ast.alias)):
            # 忽略文档字符串
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue
            # 发现非导入语句，标记为非纯导入文件
            is_pure_import = False

    return {
        'imports': imports,
        'is_pure_import': is_pure_import,
        'is_empty_content': len(content.strip()) == 0,
    }


def replace_py_file(py_filepath: str, file_info: dict, import_mapping: dict[str, str], str_mapping: dict[str, str]):
    """
    替换python文件中的内容，包括替换字符串和import语句
    Args:
        py_filepath (str): Python文件路径
        file_info (dict): Python文件解析信息
        import_mapping (Dict[str, str]): 模块路径映射字典，格式为 {旧模块路径: 新模块路径}
        str_mapping (dict[str, str]): 文本替换映射字典，格式为 {字符串: 新字符串}
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

    # 使用新的解析函数
    imports = file_info['imports'] or []
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
        for key in import_mapping:
            # 检查是否完全匹配或者是否为子模块（确保是完整的模块名前缀）
            if old_module == key or old_module.startswith(key + '.'):
                new_module = import_mapping[key] + old_module[len(key):]
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

    # 处理字符串替换
    str_mapping = str_mapping or {}
    for old_str, new_str in str_mapping.items():
        for index, line in enumerate(lines):
            if old_str in line:
                changes.append(f"Replacing '{old_str}' with '{new_str}' in {line}")
                lines[index] = line.replace(old_str, new_str)

    # 如果有变更，则写回文件
    if changes:
        with open(py_filepath, 'w', encoding='utf-8') as f:
            f.writelines(lines)
    for change in changes:
        print(change)
    return changes


def migrate_package_files(
        source_package_dir: str, target_package_dir: str,
        import_mapping: dict[str, str],
        str_mapping: dict[str, str] = None,
        include_files: list[str] = None,
        exclude_files: list[str] = None,
        clear_target=False,
        **kwargs
):
    """
    迁移包文件，将源目录中的所有文件复制到目标目录，并更新其中的导入语句

    Args:
        source_package_dir (str): 源包目录路径
        target_package_dir (str): 目标包目录路径
        import_mapping (dict[str, str]): 模块路径映射字典，格式为 {旧模块路径: 新模块路径}
        str_mapping (dict[str, str]): 文本替换映射字典，格式为 {字符串: 新字符串}
        include_files (list[str]): 包含迁移的文件列表
        exclude_files (list[str]): 排除迁移的文件列表
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
        # 获取相对于源目录的路径
        relative_file_path = file.relative_to(source_package_path).as_posix()
        # 检查是否在包含列表中
        if include_files:
            # 检查文件路径是否以包含列表中的任意项开头
            if not any(relative_file_path.startswith(include_path) for include_path in include_files):
                print(f"Skipping {file}")
                continue

        # 检查是否在忽略列表中
        if exclude_files:
            # 检查文件路径是否以忽略列表中的任意项开头
            if any(relative_file_path.startswith(ignore_path) for ignore_path in exclude_files):
                print(f"Excluding {file}")
                continue

        # 计算目标文件路径（保持相对目录结构）
        relative_path = file.relative_to(source_package_path)
        target_file = Path(target_package_dir) / relative_path

        # 确保目标文件的目录存在
        target_file.parent.mkdir(parents=True, exist_ok=True)
        if target_file.exists():
            print(f"{target_file} already exists.")
        shutil.copy2(file, target_file)  # 使用copy2保留文件元数据
        print(f"Copied {file} to {target_file}")

        # 如果是Python文件，则更新其中的导入语句
        if not str(file).endswith('.py'):
            continue

        try:
            file_info = parse_py_file(str(file))
            # 检查是否只包含import语句的py文件进行忽略
            if kwargs.get('ignore_pure_file', False) and file_info.get('is_pure_import', False):
                target_file.unlink(missing_ok=True)
                print(f"Ignoring pure import file: {file}")
                continue
            replace_py_file(str(target_file), file_info, import_mapping, str_mapping)
        except Exception as e:
            print(f"Warning: Error updating imports in {file}: {e}")

    print(f"Package migration completed: {source_package_dir} -> {target_package_dir}")
    return True


def extract_package_dependencies(package_root: str, include_prefixes=None, exclude_prefixes=None) -> dict[str, list[str]]:
    """
    扫描某个包下面所有文件import和from import依赖的包
    
    Args:
        package_root (str): 包根目录路径
        include_prefixes (list[str], optional): 包含前缀列表，只返回以这些前缀开头的模块依赖
        exclude_prefixes (list[str], optional): 排除前缀列表，排除以这些前缀开头的模块依赖
    Returns:
        dict[str, list[str]]: 依赖的包对应文件
    Example:
        # 获取所有依赖
        dependencies = extract_package_dependencies('./my_package')
        # 只获取特定前缀的依赖
        filtered_dependencies = extract_package_dependencies('./my_package', match_prefixes=['llama_index', 'langchain'])
        # 获取所有依赖但排除特定前缀的依赖
        excluded_dependencies = extract_package_dependencies('./my_package', exclude_prefixes=['torch', 'numpy'])
        # 获取特定前缀的依赖但排除某些子依赖
        mixed_dependencies = extract_package_dependencies('./my_package', 
                                                        include_prefixes=['llama_index'],
                                                        exclude_prefixes=['llama_index.legacy'])
    """
    if not os.path.isdir(package_root):
        raise ValueError(f"{package_root} is not a valid directory")

    dependency_map = {}
    package_path = Path(package_root)

    # 遍历包目录中的所有Python文件
    for py_file in package_path.rglob('*.py'):
        try:
            file_info = parse_py_file(str(py_file))
            imports = file_info.get('imports', [])

            for imp in imports:
                module_name = imp['module']

                # 跳过相对导入(带.的导入)
                if imp.get('level', 0) > 0:
                    continue

                # 检查是否需要排除
                if exclude_prefixes and any(module_name.startswith(prefix) for prefix in exclude_prefixes):
                    continue

                # 根据match_prefixes过滤模块
                if include_prefixes and any(module_name.startswith(prefix) for prefix in include_prefixes):
                    # print(f"Found dependency: {module_name} in {py_file}")
                    if module_name not in dependency_map:
                        dependency_map[module_name] = []
                    dependency_map[module_name].append(py_file)

        except Exception as e:
            print(f"Warning: Error parsing {py_file}: {e}")
            continue
    # 转换为排序后的列表返回
    return dependency_map

