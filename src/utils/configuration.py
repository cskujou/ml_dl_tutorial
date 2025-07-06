import argparse
import json
import yaml
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class BaseConfig:
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """

    @classmethod
    def load_file(cls, file_path: Path | str, override_with_args: bool = True):
        if isinstance(file_path, str):
            file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Config file {file_path} not found.")

        suffix = file_path.suffix.lower()

        if suffix == ".yaml":
            with open(file_path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
        elif suffix == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {suffix}")

        config = cls(**config_dict)
        if override_with_args:
            # 创建解析器并添加配置类的参数
            parser = argparse.ArgumentParser(description=f"{cls.__name__} Configuration")
            cls._add_config_arguments(parser)

            # 解析命令行参数
            args = parser.parse_args()

            # 用命令行参数覆盖配置
            for field in fields(cls):
                field_name = field.name
                if hasattr(args, field_name) and getattr(args, field_name) is not None:
                    # 获取字段类型
                    field_type = field.type
                    try:
                        value = getattr(args, field_name)
                        setattr(config, field_name, field_type(value))
                    except (ValueError, TypeError) as e:
                        print(f"警告: 参数 {field_name} 的值 {value} 无法转换为类型 {field_type}: {e}")
                        
        return config
                        
    @classmethod
    def _add_config_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """向解析器添加配置类的命令行参数"""
        for field in fields(cls):
            field_name = field.name
            field_type = field.type
            
            if field_type in [int, float, str, bool]:
                parser.add_argument(f'--{field_name}', type=int)
            else:
                # 对于无类型注解或其他类型，使用字符串类型
                parser.add_argument(f'--{field_name}', type=str, help=f'{field_name}')

    def save_file(self, file_path: Path | str):
        if isinstance(file_path, str):
            file_path = Path(file_path)

        suffix = file_path.suffix.lower()

        if suffix == ".yaml":
            with open(file_path, "w", encoding="utf-8") as f:
                yaml.dump(self.__dict__, f)
        elif suffix == ".json":
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.__dict__, f)
        else:
            raise ValueError(f"Unsupported config file format: {suffix}")
