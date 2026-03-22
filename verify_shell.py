from tau.tools.shell import PersistentShell
import os

def test_persistence():
    print("Testing PersistentShell...")
    shell = PersistentShell(timeout=5)
    
    # 1. Test CWD persistence
    print("\n1. Testing CWD persistence...")
    shell.execute("cd /tmp")
    cwd = shell.execute("pwd")
    print(f"  Result of 'pwd': {cwd}")
    if "/tmp" in cwd:
        print("  ✅ CWD persisted!")
    else:
        print("  ❌ CWD lost!")

    # 2. Test Environment Variable persistence
    print("\n2. Testing Env Var persistence...")
    shell.execute("export TAU_TEST_VAR=antigravity")
    val = shell.execute("echo $TAU_TEST_VAR")
    print(f"  Result of 'echo $TAU_TEST_VAR': {val}")
    if "antigravity" in val:
        print("  ✅ Env var persisted!")
    else:
        print("  ❌ Env var lost!")

    # 3. Test Exit Code capture
    print("\n3. Testing Exit Code capture...")
    res = shell.execute("ls /nonexistent_path_tau_test")
    print(f"  Result of failing command:\n{res}")
    if "[exit " in res and "[exit 0]" not in res:
        print("  ✅ Exit code captured!")
    else:
        print("  ❌ Exit code missing or zero!")

if __name__ == "__main__":
    test_persistence()
