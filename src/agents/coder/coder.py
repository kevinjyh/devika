import os
import time

from jinja2 import Environment, BaseLoader
from typing import List, Dict, Union

from src.config import Config
from src.llm import LLM
from src.state import AgentState

PROMPT = open("src/agents/coder/prompt.jinja2", "r").read().strip()

class Coder:
    def __init__(self, base_model: str):
        """
        初始化了一個 LLM 物件（這可能是一個機器學習模型），並設定了專案目錄的路徑。
        """
        config = Config()
        self.project_dir = config.get_projects_dir()
        
        self.llm = LLM(model_id=base_model)

    def render(
        self, step_by_step_plan: str, user_context: str, search_results: dict
    ) -> str:
        """
        使用 Jinja2 模板引擎來生成一個提示，該提示將被用於生成程式碼。

        參數:
            step_by_step_plan (str): 要渲染的逐步計劃。
            user_context (str): 要渲染的使用者內容。
            search_results (dict): 要渲染的搜尋結果。

        返回:
            str: 渲染後的結果。
        """
        env = Environment(loader=BaseLoader())
        template = env.from_string(PROMPT)
        return template.render(
            step_by_step_plan=step_by_step_plan,
            user_context=user_context,
            search_results=search_results,
        )

    def validate_response(self, response: str) -> Union[List[Dict[str, str]], bool]:
        """
        法驗證模型的回應是否有效。它首先將回應分割成多個部分，然後檢查每一部分是否符合預期的格式。
        如果所有部分都符合預期的格式，則該方法返回一個包含所有部分的列表；否則，它返回 False。

        Args:
            response (str): 要驗證的回應字符串。

        Returns:
            Union[List[Dict[str, str]], bool]: 如果驗證成功，返回包含文件和代碼的字典列表，否則返回 False。
        """
        response = response.strip()

        response = response.split("~~~", 1)[1]
        response = response[:response.rfind("~~~")]
        response = response.strip()

        result = []
        current_file = None
        current_code = []
        code_block = False

        for line in response.split("\n"):
            if line.startswith("File: "):
                if current_file and current_code:
                    result.append({"file": current_file, "code": "\n".join(current_code)})
                current_file = line.split("`")[1].strip()
                current_code = []
                code_block = False
            elif line.startswith("```"):
                code_block = not code_block
            else:
                current_code.append(line)

        if current_file and current_code:
            result.append({"file": current_file, "code": "\n".join(current_code)})

        return result

    def save_code_to_project(self, response: List[Dict[str, str]], project_name: str):
        """
        將程式碼儲存至專案中的指定路徑。

        Args:
            response (List[Dict[str, str]]): 包含程式碼和檔案路徑的回應列表。
            project_name (str): 專案名稱。

        Returns:
            str: 儲存程式碼的最後一個檔案路徑的目錄。
        """
        file_path_dir = None
        project_name = project_name.lower().replace(" ", "-")

        for file in response:
            file_path = f"{self.project_dir}/{project_name}/{file['file']}"
            file_path_dir = file_path[:file_path.rfind("/")]
            os.makedirs(file_path_dir, exist_ok=True)

            with open(file_path, "w") as f:
                f.write(file["code"])
    
        return file_path_dir

    def get_project_path(self, project_name: str):
        """
        返回一個專案的路徑。
        """
        project_name = project_name.lower().replace(" ", "-")
        return f"{self.project_dir}/{project_name}"

    def response_to_markdown_prompt(self, response: List[Dict[str, str]]) -> str:
        response = "\n".join([f"File: `{file['file']}`:\n```\n{file['code']}\n```" for file in response])
        return f"~~~\n{response}\n~~~"

    def emulate_code_writing(self, code_set: list, project_name: str):
        """
        模擬寫程式碼的過程。它首先獲取當前的狀態，然後創建一個新的狀態，該狀態包含了一個新的瀏覽器會話和一個新的終端會話。
        然後，它將新的狀態添加到當前的狀態中，並等待一段時間。

        Args:
            code_set (list): 包含要寫入的程式碼的清單。
            project_name (str): 專案名稱。

        Returns:
            None
        """
        for current_file in code_set:
            file = current_file["file"]
            code = current_file["code"]

            current_state = AgentState().get_latest_state(project_name)
            new_state = AgentState().new_state()
            new_state["browser_session"] = current_state["browser_session"] # 保留瀏覽器會話
            new_state["internal_monologue"] = "正在寫程式碼..."
            new_state["terminal_session"]["title"] = f"編輯 {file}"
            new_state["terminal_session"]["command"] = f"vim {file}"
            new_state["terminal_session"]["output"] = code
            AgentState().add_to_current_state(project_name, new_state)
            time.sleep(2)

    def execute(
            self,
            step_by_step_plan: str,
            user_context: str,
            search_results: dict,
            project_name: str
        ) -> str:
        """
        此為類別的主要方法。它首先生成一個提示，然後使用 LLM 物件來生成一個回應。
        然後，它驗證該回應是否有效，並將其保存到一個專案中。
        最後，它模擬寫程式碼的過程，並返回模型的回應。

        Args:
            step_by_step_plan (str): The step-by-step plan for coding.
            user_context (str): The user context for coding.
            search_results (dict): The search results for coding.
            project_name (str): The name of the project.

        Returns:
            str: The valid response from the coding process.
        """
        prompt = self.render(step_by_step_plan, user_context, search_results)
        response = self.llm.inference(prompt)
        
        valid_response = self.validate_response(response)
        
        while not valid_response:
            print("Invalid response from the model, trying again...")
            return self.execute(step_by_step_plan, user_context, search_results)
        
        print(valid_response)
        
        self.emulate_code_writing(valid_response, project_name)

        return valid_response
