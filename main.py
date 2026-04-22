"""
main.py — CLI entrypoint for the AutoStream AI Agent.
"""
import sys
from dotenv import load_dotenv

def main():
    load_dotenv()
    
    try:
        from agent import chat, load_kb
        import agent
    except ImportError as e:
        print(f"[ERROR] Failed to import modules: {e}")
        sys.exit(1)

    try:
        kb = load_kb()
    except Exception as e:
        print(f"[ERROR] Failed to load knowledge base: {e}")
        sys.exit(1)

    print("""\
╔══════════════════════════════╗
║   AutoStream AI Assistant    ║
║   Type 'exit' to quit        ║
╚══════════════════════════════╝""")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye! 👋")
            break
            
        if user_input.lower() in ("exit", "quit", "q", "/quit", "/exit"):
            print("Bye! 👋")
            break
            
        if not user_input:
            continue
            
        response = chat(user_input, kb)
        
        # If the response wasn't already streamed directly to console by get_agent_response
        if not agent.was_streamed:
            print(f"Maya: {response}")

if __name__ == "__main__":
    main()
