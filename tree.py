import os

# ================= ⚙️ 配置区域 =================
OUTPUT_FILE = "project_structure.txt"

# 🚫 忽略的文件夹
IGNORE_DIRS = {'.vscode', '.git', '__pycache__', '.idea', 'venv', 'node_modules', 'labels'}

# 🚫 忽略的文件后缀
IGNORE_EXTS = {'.png', '.jpg', '.jpeg', '.gif', '.pyc', '.exe', '.dll'}

# 🎨 图标映射表 (在这里添加你想要的图标)
ICON_MAP = {
    # 编程语言
    '.py': '🐍',     # Python
    '.js': '🟨',     # JavaScript
    '.ts': '🔷',     # TypeScript
    '.html': '🌐',   # HTML
    '.css': '🎨',    # CSS
    '.java': '☕',   # Java
    '.c': '🇨',      # C
    '.cpp': '➕',    # C++
    '.go': '🐹',     # Go
    '.sh': '🐚',     # Shell Script

    # 数据与配置
    '.json': '⚙️ ',   # JSON
    '.yaml': '⚙️ ',   # YAML
    '.yml': '⚙️ ',    # YAML
    '.xml': '📰',    # XML
    '.ini': '🔧',    # INI
    '.env': '🔒',    # Env variables

    # 文档与文本
    '.md': '📘',     # Markdown
    '.txt': '📝',    # Text
    '.pdf': '📕',    # PDF
    '.csv': '📊',    # CSV

    # 其他
    '.zip': '📦',    # Archive
    '.gitignore': '🙈' # Git ignore
}

# 📁 默认图标
DEFAULT_FILE_ICON = '📄'
DEFAULT_FOLDER_ICON = '📂'
# ===========================================

def get_file_icon(filename):
    """根据后缀名获取图标"""
    # 处理特殊文件名，如 .gitignore
    if filename == '.gitignore':
        return ICON_MAP.get('.gitignore')
    
    _, ext = os.path.splitext(filename)
    # 查找映射表，找不到则返回默认图标
    return ICON_MAP.get(ext.lower(), DEFAULT_FILE_ICON)

def generate_tree(startpath, file_handle):
    print(f"🚀 开始扫描: {startpath}")
    
    for root, dirs, files in os.walk(startpath):
        # 1. 过滤文件夹 (原地修改 dirs 列表)
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        # 计算层级缩进
        rel_path = root.replace(startpath, '').lstrip(os.sep)
        if rel_path == '':
            level = 0
        else:
            level = rel_path.count(os.sep) + 1
            
        indent = '    ' * level
        
        # 打印文件夹名
        folder_name = os.path.basename(root)
        if folder_name == '': folder_name = os.path.basename(startpath) # 根目录名
        
        folder_line = f"{indent}{DEFAULT_FOLDER_ICON} {folder_name}/\n"
        file_handle.write(folder_line)
        
        # 2. 过滤并打印文件
        subindent = '    ' * (level + 1)
        for f in files:
            _, ext = os.path.splitext(f)
            
            # 检查是否忽略
            if ext.lower() not in IGNORE_EXTS:
                # 获取美化图标
                icon = get_file_icon(f)
                file_line = f"{subindent}{icon} {f}\n"
                file_handle.write(file_line)

if __name__ == "__main__":
    # 获取当前脚本所在目录
    root_dir = os.getcwd()
    
    print(f"正在生成目录树到 {OUTPUT_FILE} ...")
    
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            # 写入一个标题
            f.write(f"Project Structure: {os.path.basename(root_dir)}\n")
            f.write("=" * 30 + "\n")
            generate_tree(root_dir, f)
            
        print(f"\n✅ 成功！文件已保存为: {OUTPUT_FILE}")

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")