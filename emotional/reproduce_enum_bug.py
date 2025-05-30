# reproduce_enum_bug.py
import unittest
import warnings

# 尝试导入必要的 transformers 模块
try:
    from transformers.utils.quantization_config import (
        QuantizationMethod,
        QuantizationConfig,
        register_quantization_config,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    # 如果 transformers 不可用，测试将被跳过
    # 定义一些模拟对象，以便文件本身可以运行（尽管测试不会执行）
    class QuantizationMethod: # type: ignore
        _value2member_map_ = {}
        def __init__(self, value): raise ValueError("模拟对象") # Mock
    class QuantizationConfig: pass # type: ignore
    def register_quantization_config(name, config_class): pass # type: ignore

if TRANSFORMERS_AVAILABLE:
    class MyCustomBugReproQuantConfig(QuantizationConfig):
        # 这是一个用于测试注册的虚拟配置类
        pass

@unittest.skipIf(not TRANSFORMERS_AVAILABLE, "未找到 Transformers 库或其量化工具模块。")
class TestEnumUpdateBug(unittest.TestCase):
    def test_quantization_method_enum_not_updated(self):
        """
        测试在调用 register_quantization_config 后，QuantizationMethod 枚举是否已更新。
        如果 bug (issue #38462) 存在，此测试应该会失败 (FAIL)，
        因为枚举将不包含新方法。
        """
        new_method_name = "my_very_custom_method_for_repro" # 用于复现的非常自定义的方法名

        # 可选：如果在同一会话中由于先前运行而已存在此名称，则进行清理
        # 对于枚举来说这比较困难，因此使用唯一的名称更好。
        if new_method_name in QuantizationMethod._value2member_map_:
            warnings.warn(
                f"警告：测试方法 '{new_method_name}' 已存在于 QuantizationMethod 中。 "
                "如果不是干净的环境，这可能会影响测试的有效性。"
            )

        print(f"\n尝试注册: '{new_method_name}'")
        print(f"注册前的 QuantizationMethod 值: {list(QuantizationMethod._value2member_map_.keys())}")

        # 操作：注册新的量化方法
        register_quantization_config(new_method_name, MyCustomBugReproQuantConfig)

        print(f"注册后的 QuantizationMethod 值: {list(QuantizationMethod._value2member_map_.keys())}")

        # 断言：检查新方法是否存在于枚举的值映射中
        # 如果 bug 存在，此断言预计会失败 (FAIL)。
        self.assertIn(
            new_method_name,
            QuantizationMethod._value2member_map_,
            f"BUG 已复现：'{new_method_name}' 未能添加到 QuantizationMethod._value2member_map_ 中。"
        )

        # 此外，如果已注册，尝试通过枚举的构造函数访问它应该能成功。
        # 如果未找到，则会引发 ValueError，这也表明存在 bug。
        try:
            _ = QuantizationMethod(new_method_name)
            print(f"成功 (bug 已修复？): QuantizationMethod('{new_method_name}') 已找到。")
        except ValueError:
            self.fail(
                f"BUG 已复现 (通过 ValueError)：无法创建 QuantizationMethod('{new_method_name}')， "
                "确认它不在枚举中。"
            )

if __name__ == "__main__":
    print(f"Transformers 可用于测试: {TRANSFORMERS_AVAILABLE}")
    if TRANSFORMERS_AVAILABLE:
         print(f"初始 QuantizationMethod 值: {list(QuantizationMethod._value2member_map_.keys())}")
    unittest.main(verbosity=2)