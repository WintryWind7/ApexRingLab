"""毒圈数据爬虫"""

import requests
import json
from pathlib import Path
from typing import Optional, Dict, Any, Literal
from datetime import datetime

# ==================== 配置区域 ====================
# ALGS 模式配置
ALGS_TOKEN = "e53ce4adccd9234eaa94283601652799"
ALGS_COOKIE = 'apexlegendsstatus_ssid=m3tm9b8femgdg9e352o696ab15; _ga=GA1.1.187038249.1768132477; cc_cookie={"level":["necessary","recaptcha"],"revision":0,"data":null,"rfc_cookie":false}; als_lang=en; cf_clearance=24QKG5pbFjqjOF.xufng9zg9ToWEgHXeV4mm28oFG40-1768385903-1.2.1.1-5Pzn3cPOutfC4iYnzYIB5_628NjzjshUX87ojQeh4qV4IZzlFLjScKRsQ80dIVTOuxuq9EevF2eGThvYzowjGdw.RBCZJR.lMIWHI71O.UdTJR40AgzygQLSWNSqcr_5n9rV91nBrohXAMTfNi.04rWqSVWPewNzrGluCJf7rOV80CPpdxl5GfB8oeYexwb81IvfloY.gyRSNL0t5Mau08GU8OgIk1EhKNTjIxUdosE; _ga_HGQ7L9V5FY=GS2.1.s1768385899$o5$g1$t1768386220$j60$l0$h0; dgs_token=QUepnP%2BrLwfB4cPItWKOwfFUy42DI3VSiuN%2BqyqdkUh2Mcu%2FiTLMKuF01054nvN87kAQXKfnzZvv4AcHVi2CkNxsTNCK437pWDUW50pmu%2FbNgi34RGYRXuSAj%2FtHyb5gjJvszNbd8ZAFOj95Dpuy2e5lO6xntNBwo%2F31mecUjDaxGYEcjLqdTGT1hfmPXM0Of%2BQnpkg9FyNrJZuwGPC3Qs%2BrNHXL3SN1iIVGTiL%2FLNsTsVxv2ufHa5e2gOuS%2BROR%2FL8%2FGXKd%2Bf%2FwN635t5cFXjCycglKZ9WZZCxRIfkZKSSrXm9RhNhOaypUkjLPp8GPGSaBRXJNSBjEV6UrYi0veHuYOPKyQntSPyES7PQnSwpPbEP3K3lMptfZJm%2BfKe5O%2FFK7f3zUXNTLsOOhRHY7fhgFh1kXgXy99W%2BhsIrQPhhZOCoyZm2p4wphPA3jJNuEQMBW%2F6FJNShGHXdlxYjZskLt1DA5T34%3D'

# PUBLIC 模式配置
PUBLIC_TOKEN = "97989a26d77b956c000c5f48f3e33617"
PUBLIC_COOKIE = 'apexlegendsstatus_ssid=m3tm9b8femgdg9e352o696ab15; _ga=GA1.1.187038249.1768132477; cc_cookie={"level":["necessary","recaptcha"],"revision":0,"data":null,"rfc_cookie":false}; als_lang=en; cf_clearance=24QKG5pbFjqjOF.xufng9zg9ToWEgHXeV4mm28oFG40-1768385903-1.2.1.1-5Pzn3cPOutfC4iYnzYIB5_628NjzjshUX87ojQeh4qV4IZzlFLjScKRsQ80dIVTOuxuq9EevF2eGThvYzowjGdw.RBCZJR.lMIWHI71O.UdTJR40AgzygQLSWNSqcr_5n9rV91nBrohXAMTfNi.04rWqSVWPewNzrGluCJf7rOV80CPpdxl5GfB8oeYexwb81IvfloY.gyRSNL0t5Mau08GU8OgIk1EhKNTjIxUdosE; _ga_HGQ7L9V5FY=GS2.1.s1768385899$o5$g1$t1768386220$j60$l0$h0; dgs_token=QUepnP%2BrLwfB4cPItWKOwfFUy42DI3VSiuN%2BqyqdkUh2Mcu%2FiTLMKuF01054nvN87kAQXKfnzZvv4AcHVi2CkNxsTNCK437pWDUW50pmu%2FbNgi34RGYRXuSAj%2FtHyb5gjJvszNbd8ZAFOj95Dpuy2e5lO6xntNBwo%2F31mecUjDaxGYEcjLqdTGT1hfmPXM0Of%2BQnpkg9FyNrJZuwGPC3Qs%2BrNHXL3SN1iIVGTiL%2FLNsTsVxv2ufHa5e2gOuS%2BROR%2FL8%2FGXKd%2Bf%2FwN635t5cFXjCycglKZ9WZZCxRIfkZKSSrXm9RhNhOaypUkjLPp8GPGSaBRXJNSBjEV6UrYi0veHuYOPKyQntSPyES7PQnSwpPbEP3K3lMptfZJm%2BfKe5O%2FFK7f3zUXNTLsOOhRHY7fhgFh1kXgXy99W%2BhsIrQPhhZOCoyZm2p4wphPA3jJNuEQMBW%2F6FJNShGHXdlxYjZskLt1DA5T34%3D'

# 爬取配置
REQUEST_DELAY = 0.2  # 每次请求间隔（秒）
MAX_CONSECUTIVE_FAILS = 3  # 最大连续失败次数
# ==================================================


class RingCrawler:
    """毒圈数据爬虫类"""
    
    def __init__(
        self, 
        mode: Literal["ALGS", "PUBLIC"] = "PUBLIC",
        token: Optional[str] = None,
        cookie: Optional[str] = None,
        output_dir: Optional[str] = None
    ):
        """
        初始化爬虫
        
        Args:
            mode: 爬取模式，ALGS 或 PUBLIC
            token: API 访问 token，不提供则使用配置区域的默认值
            cookie: 浏览器 cookie，不提供则使用配置区域的默认值
            output_dir: 数据保存文件路径，不提供则根据模式自动设置
        """
        self.mode = mode
        
        # 根据模式设置 token 和 cookie
        if mode == "ALGS":
            self.token = token or ALGS_TOKEN
            self.cookie = cookie or ALGS_COOKIE
            default_output_file = "data/crawler/algs_rings.jsonl"
        else:  # PUBLIC
            self.token = token or PUBLIC_TOKEN
            self.cookie = cookie or PUBLIC_COOKIE
            default_output_file = "data/crawler/public_rings.jsonl"
        
        # 设置输出文件（相对于项目根目录）
        if output_dir:
            self.output_file = Path(output_dir)
        else:
            # 获取项目根目录（ring_crawler.py 在 utils/ 下）
            project_root = Path(__file__).parent.parent
            self.output_file = project_root / default_output_file
        
        # 创建父目录
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.api_url = "https://apexlegendsstatus.com/algs/ringguesser/api/getRandomRing"
        self.headers = {
            "accept": "*/*",
            "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
            "cookie": self.cookie,
            "referer": "https://apexlegendsstatus.com/algs/ring-guesser",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36"
        }
        
        # 初始化去重集合
        self.data_hashes = set()
        self._load_existing_data()
    
    def _load_existing_data(self) -> None:
        """加载已有数据到内存，用于去重"""
        if not self.output_file.exists():
            return
        
        try:
            with open(self.output_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        data_hash = self._compute_hash(data)
                        self.data_hashes.add(data_hash)
            
            print(f"已加载 {len(self.data_hashes)} 条已有数据用于去重")
        except Exception as e:
            print(f"加载已有数据失败: {e}")
    
    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """
        计算数据的 hash 值（排除 timestamp 字段）
        
        Args:
            data: 数据字典
            
        Returns:
            数据的 hash 字符串
        """
        # 复制数据并移除 timestamp
        data_copy = data.copy()
        data_copy.pop("timestamp", None)
        
        # 转为 JSON 字符串并计算 hash
        import hashlib
        data_str = json.dumps(data_copy, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _is_duplicate(self, data: Dict[str, Any]) -> bool:
        """
        检查数据是否重复
        
        Args:
            data: 要检查的数据
            
        Returns:
            重复返回 True，否则返回 False
        """
        data_hash = self._compute_hash(data)
        return data_hash in self.data_hashes
    
    def fetch_ring_data(self) -> Optional[Dict[str, Any]]:
        """
        获取一次毒圈数据
        
        Returns:
            成功返回数据字典，失败返回 None
        """
        params = {
            "mode": self.mode,
            "token": self.token
        }
        
        try:
            response = requests.get(
                self.api_url,
                params=params,
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            # 验证数据完整性
            if self._validate_data(data):
                return data
            else:
                print(f"数据验证失败: {data}")
                return None
                
        except Exception as e:
            print(f"请求失败: {e}")
            return None
    
    def _validate_data(self, data: Dict[str, Any]) -> bool:
        """
        验证数据完整性
        
        Args:
            data: API 返回的数据
            
        Returns:
            数据有效返回 True，否则返回 False
        """
        if not isinstance(data, dict):
            return False
        
        # 必须包含 map 和 rings
        if "map" not in data or "rings" not in data:
            return False
        
        # map 不能为空
        if not data["map"]:
            return False
        
        # rings 必须是列表且不为空
        if not isinstance(data["rings"], list) or len(data["rings"]) == 0:
            return False
        
        # 验证每个圈的数据结构
        for ring in data["rings"]:
            if not isinstance(ring, dict):
                return False
            if "x" not in ring or "y" not in ring or "r" not in ring:
                return False
        
        return True
    
    def save_data(self, data: Dict[str, Any]) -> bool:
        """
        保存数据到 JSONL 文件（自动去重）
        
        Args:
            data: 要保存的数据
            
        Returns:
            保存成功返回 True，重复或失败返回 False
        """
        try:
            # 检查是否重复
            if self._is_duplicate(data):
                return False
            
            # 添加时间戳
            data["timestamp"] = datetime.now().isoformat()
            
            # 追加到文件
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
            
            # 添加到去重集合
            data_hash = self._compute_hash(data)
            self.data_hashes.add(data_hash)
            
            return True
            
        except Exception as e:
            print(f"保存失败: {e}")
            return False
    
    def crawl_once(self) -> bool:
        """
        爬取并保存一次数据
        
        Returns:
            成功返回 True，失败返回 False
        """
        data = self.fetch_ring_data()
        if data:
            return self.save_data(data)
        return False
    
    def crawl_until_fail(self, delay: Optional[float] = None, max_consecutive_fails: Optional[int] = None) -> int:
        """
        持续爬取直到失败
        
        Args:
            delay: 每次请求间隔（秒），不提供则使用配置区域的默认值
            max_consecutive_fails: 最大连续失败次数，不提供则使用配置区域的默认值
            
        Returns:
            成功爬取的数量
        """
        import time
        
        delay = delay or REQUEST_DELAY
        max_consecutive_fails = max_consecutive_fails or MAX_CONSECUTIVE_FAILS
        
        success_count = 0
        duplicate_count = 0
        consecutive_fails = 0
        total_attempts = 0
        
        print(f"开始持续爬取，间隔 {delay} 秒...")
        print(f"连续失败 {max_consecutive_fails} 次后将停止\n")
        
        while consecutive_fails < max_consecutive_fails:
            total_attempts += 1
            print(f"第 {total_attempts} 次尝试...", end=" ")
            
            data = self.fetch_ring_data()
            if data:
                if self.save_data(data):
                    success_count += 1
                    consecutive_fails = 0
                    print(f"✓ 新数据 (不重复: {success_count}, 重复: {duplicate_count})")
                else:
                    duplicate_count += 1
                    consecutive_fails = 0
                    print(f"⊙ 重复 (不重复: {success_count}, 重复: {duplicate_count})")
            else:
                consecutive_fails += 1
                print(f"✗ 失败 (连续失败: {consecutive_fails}/{max_consecutive_fails})")
            
            # 如果还没达到停止条件，等待后继续
            if consecutive_fails < max_consecutive_fails:
                time.sleep(delay)
        
        print(f"\n爬取结束！")
        print(f"总尝试: {total_attempts} 次")
        print(f"新数据: {success_count} 条")
        print(f"重复: {duplicate_count} 条")
        print(f"失败: {total_attempts - success_count - duplicate_count} 次")
        return success_count
        """
        批量爬取数据
        
        Args:
            count: 爬取次数
            delay: 每次请求间隔（秒）
            
        Returns:
            成功爬取的数量
        """
        import time
        
        success_count = 0
        for i in range(count):
            print(f"正在爬取第 {i+1}/{count} 条数据...")
            
            if self.crawl_once():
                success_count += 1
                print(f"✓ 成功")
            else:
                print(f"✗ 失败")
            
            # 最后一次不需要延迟
            if i < count - 1:
                time.sleep(delay)
        
        print(f"\n完成！成功: {success_count}/{count}")
        return success_count


if __name__ == "__main__":
    # 使用示例
    
    # 方式 1: 使用配置区域的默认值
    # PUBLIC 模式
    crawler_public = RingCrawler(mode="PUBLIC")
    print("=== PUBLIC 模式 ===")
    crawler_public.crawl_until_fail()
    
    # ALGS 模式（需要先更新顶部配置区域的 ALGS_TOKEN）
    # crawler_algs = RingCrawler(mode="ALGS")
    # print("\n=== ALGS 模式 ===")
    # crawler_algs.crawl_until_fail()
    
    # 方式 2: 手动指定参数
    # crawler = RingCrawler(
    #     mode="PUBLIC",
    #     token="your_token_here",
    #     cookie="your_cookie_here",
    #     output_dir="data/custom_rings.jsonl"
    # )
    # crawler.crawl_until_fail(delay=5.0, max_consecutive_fails=5)
