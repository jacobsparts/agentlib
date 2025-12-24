import unittest

class TestImports(unittest.TestCase):
    def test_imports(self):
        """Test that all public modules can be imported"""
        from agentlib import BaseAgent
        from agentlib.client import LLMClient
        from agentlib.conversation import Conversation
        from agentlib.utils import JSON_INDENT
        
        # Verify the imports worked
        self.assertIsNotNone(BaseAgent)
        self.assertIsNotNone(LLMClient)
        self.assertIsNotNone(Conversation)
        # JSON_INDENT is intentionally None for compact output

if __name__ == "__main__":
    unittest.main()
