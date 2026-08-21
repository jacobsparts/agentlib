import unittest

class TestImports(unittest.TestCase):
    def test_imports(self):
        """Test that all public modules can be imported"""
        from agentlib import BaseAgent, one_shot, REPLAgent, REPLAgentLegacy
        from agentlib.client import LLMClient
        from agentlib.client import _gemini_transform_schema
        from agentlib.conversation import Convo
        from agentlib.utils import JSON_INDENT
        
        # Verify the imports worked
        self.assertIsNotNone(BaseAgent)
        self.assertIsNotNone(one_shot)
        self.assertIsNotNone(REPLAgent)
        self.assertIsNotNone(REPLAgentLegacy)
        self.assertIsNotNone(LLMClient)
        self.assertIsNotNone(_gemini_transform_schema)
        self.assertIsNotNone(Convo)
        # JSON_INDENT is intentionally None for compact output

if __name__ == "__main__":
    unittest.main()
