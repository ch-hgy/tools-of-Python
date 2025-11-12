import json
import os
import csv

class SaveDict:
    @staticmethod
    def save_dict(filename, data_dict, cover=False):
        """
        保存字典到文件
        
        Args:
            filename (str): 文件名
            data_dict (dict): 需要保存的字典
            cover (bool): 是否清空原文件覆写，False则追加到文件末尾
        """
        if cover or not os.path.exists(filename):
            # 覆盖模式或文件不存在，直接写入
            mode = 'w'
            data_to_save = data_dict
        else:
            # 追加模式，读取现有内容并合并
            mode = 'r+'
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                
                # 合并字典
                if isinstance(existing_data, list):
                    # 如果原数据是列表，追加新字典
                    existing_data.append(data_dict)
                    data_to_save = existing_data
                elif isinstance(existing_data, dict):
                    # 如果原数据是字典，创建包含两个字典的列表
                    data_to_save = [existing_data, data_dict]
                else:
                    data_to_save = [existing_data, data_dict]
                    
            except (json.JSONDecodeError, FileNotFoundError):
                data_to_save = data_dict
                mode = 'w'
        
        with open(filename, mode, encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)
    
    @staticmethod
    def read_dict(filename):
        """
        从文件读取字典
        
        Args:
            filename (str): 文件名
            
        Returns:
            dict or list: 读取到的数据，如果文件不存在返回空字典
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        
    
class SaveList:
    def __init__(self):
        pass
    
    def save_list(self, filename, data_list, cover=False):
        """
        将列表存储到CSV文件
        
        Args:
            filename (str): 文件名
            data_list (list): 需要保存的列表
            cover (bool): 是否清空原文件覆写，False则追加新行
        """
        # 如果是覆盖模式或文件不存在，创建新文件
        if cover or not os.path.exists(filename):
            mode = 'w'
        else:
            mode = 'a'
        
        # 保存数据
        with open(filename, mode, newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(data_list)
    
    def read_list(self, filename, index=None):
        """
        从CSV文件读取列表
        
        Args:
            filename (str): 文件名
            index (int, optional): 指定读取第几行，None则返回所有行
            
        Returns:
            list or list of lists: 读取到的列表数据
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                all_lists = []
                
                for row in reader:
                    # 转换数据类型（尝试转换为数字，失败则保持字符串）
                    converted_row = []
                    for item in row:
                        try:
                            # 尝试转换为整数
                            converted_row.append(int(item))
                        except ValueError:
                            try:
                                # 尝试转换为浮点数
                                converted_row.append(float(item))
                            except ValueError:
                                # 保持为字符串
                                converted_row.append(item)
                    all_lists.append(converted_row)
            
            if index is not None:
                return all_lists[index] if 0 <= index < len(all_lists) else None
            else:
                return all_lists
                
        except FileNotFoundError:
            print(f"文件 {filename} 不存在")
            return []
        except Exception as e:
            print(f"读取文件时出错: {e}")
            return []
    
    def clear_file(self, filename):
        """
        清空文件内容
        """
        open(filename, 'w').close()  # 清空文件内容
    
    def get_count(self, filename):
        """
        获取文件中保存的列表数量（行数）
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                return sum(1 for _ in reader)
        except FileNotFoundError:
            return 0