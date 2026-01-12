"""毒圈数据爬虫"""

import requests
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class RingCrawler:
    """毒圈数据爬虫类"""
    
    def __init__(self, token: str, cookie: str, output_dir: str = "data/rings_3"):
        """
        初始化爬虫
        
        Args:
            token: API 访问 token
            cookie: 浏览器 cookie
            output_dir: 数据保存目录
        """
        self.token = token
        self.cookie = cookie
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / "rings_data.jsonl"
        
        self.api_url = "https://apexlegendsstatus.com/algs/ringguesser/api/getRandomRing"
        self.headers = {
            "accept": "*/*",
            "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
            "cookie": cookie,
            "referer": "https://apexlegendsstatus.com/algs/ring-guesser",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36"
        }
    
    def fetch_ring_data(self) -> Optional[Dict[str, Any]]:
        """
        获取一次毒圈数据
        
        Returns:
            成功返回数据字典，失败返回 None
        """
        params = {
            "mode": "ALGS",
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
        保存数据到 JSONL 文件
        
        Args:
            data: 要保存的数据
            
        Returns:
            保存成功返回 True，失败返回 False
        """
        try:
            # 添加时间戳
            data["timestamp"] = datetime.now().isoformat()
            
            # 追加到文件
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
            
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
    
    def crawl_until_fail(self, delay: float = 3.0, max_consecutive_fails: int = 3) -> int:
        """
        持续爬取直到失败
        
        Args:
            delay: 每次请求间隔（秒）
            max_consecutive_fails: 最大连续失败次数，达到后停止
            
        Returns:
            成功爬取的数量
        """
        import time
        
        success_count = 0
        consecutive_fails = 0
        total_attempts = 0
        
        print(f"开始持续爬取，间隔 {delay} 秒...")
        print(f"连续失败 {max_consecutive_fails} 次后将停止\n")
        
        while consecutive_fails < max_consecutive_fails:
            total_attempts += 1
            print(f"第 {total_attempts} 次尝试...", end=" ")
            
            if self.crawl_once():
                success_count += 1
                consecutive_fails = 0
                print(f"✓ 成功 (总成功: {success_count})")
            else:
                consecutive_fails += 1
                print(f"✗ 失败 (连续失败: {consecutive_fails}/{max_consecutive_fails})")
            
            # 如果还没达到停止条件，等待后继续
            if consecutive_fails < max_consecutive_fails:
                time.sleep(delay)
        
        print(f"\n爬取结束！")
        print(f"总尝试: {total_attempts} 次")
        print(f"成功: {success_count} 次")
        print(f"失败: {total_attempts - success_count} 次")
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
    # 从浏览器开发者工具复制 cookie 和 token
    TOKEN = "d19891c3f15dcc59a60935bb758eb043"
    COOKIE = 'apexlegendsstatus_ssid=m3tm9b8femgdg9e352o696ab15; _ga=GA1.1.187038249.1768132477; cc_cookie={"level":["necessary","recaptcha"],"revision":0,"data":null,"rfc_cookie":false}; als_lang=en; _ga_HGQ7L9V5FY=GS2.1.s1768198759$o4$g0$t1768198763$j56$l0$h0; cf_clearance=p.jvapCuKI0fA4M9lcBYEFNpQpzuijk7t1XJcDFKNKw-1768204677-1.2.1.1-dhVyGyTM2MuFL3sktCr4UDl7x_rXyVvdt94Wyo1_s9Zr6V6e6lLgoDu3QI2ZpZJXvijJw6zCtQQHBUmsf06j8etkP9wQBuHEVRWWfJMTgcoWhhEMnwCse8_5jRJ9ZBT354egWk.DGPfdn9dyVUt7iJ9C0LFCWfafIjrgsAlTCnqDSDRBY3MRWEQPl1dH.kdr4XMb9fXwQHChJw6iUFLa97fa7w8ojpT93kAIwrBJ6OzKACcoIfeaqgYUloIIfv2H; dgs_token=tpQEe3cBVHbBJsuLRnDdo35g4N8XDaVvxc8K4ZQVA%2FANYPX3tr96E0VCTQR%2BNNkAr6m%2Bk96DDS3bmNFeAVblc62RmIXpRGMk1RXdXSOwaVcgJR3hQi9krCbpt%2BBvPIQWfiHi%2F%2FeqriWv%2B4qExTPZku%2F2HCFLZkMRvBuS%2FtwErQe2ckhnCOcn%2BIlTzrAPXc6qcvBCJVcjDLYP%2FLMITLPHkxMD9xIqp%2Fl%2Bo5%2Bin45Z5Jrq867tw3KaT7DN059WtTZljBo565%2BMZhLUbk%2F41TgmvoTkjtq%2FdKZWqhIbh4SbAk1iZZEaD5hy4RjlEYlCO2A6zvsFMlMFTPSplP%2Fo8b37g9mWITlCzMOpYoox%2F27rNHdnPcRYo5%2FYHFYY1J%2BxXdus5bnQeeb99%2FdKrcSF5Dehgd4j%2Bsf42zzjLvDKUXDVlLCyRswK9tIjnGGAkY6Qo%2Fw403Z9J9pjfvhfEWDlj6VvnEYDfn8IJXM%3D'
    
    crawler = RingCrawler(token=TOKEN, cookie=COOKIE)
    
    # 持续爬取直到失败
    crawler.crawl_until_fail(delay=3.0)
