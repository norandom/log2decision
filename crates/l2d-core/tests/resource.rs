use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use l2d_core::resource::{BusyChoice, ResourceGuard};

#[test]
fn resource_child() {
    let Ok(dir) = std::env::var("L2D_TEST_GATE_DIR") else {
        return;
    };
    if std::env::var("L2D_TEST_GATE_CRASH").is_ok() {
        let _guard =
            ResourceGuard::at(Path::new(&dir), "crashed-process", BusyChoice::Refuse, true)
                .unwrap();
        // Exit bypasses Rust destructors, like termination by the host.
        std::process::exit(64);
    }
    let choice = if std::env::var("L2D_TEST_GATE_WAIT").is_ok() {
        BusyChoice::Wait
    } else {
        BusyChoice::Refuse
    };
    let result = ResourceGuard::at(Path::new(&dir), "second-process", choice, true);
    match result {
        Ok(guard) => {
            std::fs::write(Path::new(&dir).join("entered"), guard.waited.to_string()).unwrap();
            assert!(
                choice == BusyChoice::Wait,
                "second process entered an occupied host"
            );
        }
        Err(error) => {
            assert_eq!(choice, BusyChoice::Refuse, "{error}");
            assert!(error.to_string().contains("first-process"), "{error}");
        }
    }
}

fn child(dir: &Path, wait: bool) -> std::process::Child {
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args(["--exact", "resource_child", "--nocapture"])
        .env("L2D_TEST_GATE_DIR", dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if wait {
        command.env("L2D_TEST_GATE_WAIT", "1");
    }
    command.spawn().unwrap()
}

fn finish(mut child: std::process::Child) {
    let start = Instant::now();
    while child.try_wait().unwrap().is_none() {
        if start.elapsed() > Duration::from_secs(10) {
            child.kill().unwrap();
            let output = child.wait_with_output().unwrap();
            panic!("child hung: {}", String::from_utf8_lossy(&output.stderr));
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn actual_second_process_refuses_or_waits_and_nested_npu_then_fill_does_not_deadlock() {
    let dir = std::env::temp_dir().join(format!("l2d-gate-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let first = ResourceGuard::at(&dir, "first-process", BusyChoice::Refuse, true).unwrap();
    assert!(first.observation.free_memory_bytes > 0);
    assert_eq!(first.observation.external_npu_utilization, "unknown");
    let nested = ResourceGuard::at(&dir, "fill", BusyChoice::Refuse, false).unwrap();
    assert!(!nested.waited);
    finish(child(&dir, false));
    assert!(!dir.join("entered").exists());
    let waiting = child(&dir, true);
    std::thread::sleep(Duration::from_millis(150));
    assert!(!dir.join("entered").exists());
    drop(first);
    // A nested owner keeps the OS lock alive even if the outer scope ends first.
    finish(child(&dir, false));
    drop(nested);
    finish(waiting);
    assert_eq!(
        std::fs::read_to_string(dir.join("entered")).unwrap(),
        "true"
    );
    // Stale owner metadata is ignored after the OS has released the actual lock.
    let next = ResourceGuard::at(&dir, "after-exit", BusyChoice::Refuse, true).unwrap();
    assert!(next.observation.host_holder.is_none());
    drop(next);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_crashed_holder_does_not_leave_the_host_or_npu_locked() {
    let dir = std::env::temp_dir().join(format!("l2d-gate-crash-{}", std::process::id()));
    let status = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "resource_child"])
        .env("L2D_TEST_GATE_DIR", &dir)
        .env("L2D_TEST_GATE_CRASH", "1")
        .stdout(Stdio::null())
        .status()
        .unwrap();
    assert_eq!(status.code(), Some(64));
    let guard = ResourceGuard::at(&dir, "after-crash", BusyChoice::Refuse, true).unwrap();
    assert!(guard.observation.host_holder.is_none());
    assert!(guard.observation.npu_holder.is_none());
    drop(guard);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn first_acquisition_waits_for_atomic_publication_and_rejects_stale_metadata() {
    let dir = std::env::temp_dir().join(format!("l2d-gate-publication-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let host = std::fs::File::create(dir.join("host.lock")).unwrap();
    host.try_lock().unwrap();
    // Simulate a crash's metadata: its witness is no longer locked. A contender must
    // neither name that stale owner nor treat initialization as a malformed lock.
    std::fs::File::create(dir.join("stale.lock")).unwrap();
    std::fs::write(
        dir.join("host.holder.json"),
        br#"{"process_id":1,"operation":"stale","lease_file":"stale.lock"}"#,
    )
    .unwrap();
    let mut waiting = child(&dir, true);
    std::thread::sleep(Duration::from_millis(200));
    assert!(
        waiting.try_wait().unwrap().is_none(),
        "publication gap must wait"
    );
    let witness = std::fs::File::create(dir.join("live.lock")).unwrap();
    witness.try_lock().unwrap();
    std::fs::write(
        dir.join("publishing"),
        br#"{"process_id":1,"operation":"first-process","lease_file":"live.lock"}"#,
    )
    .unwrap();
    std::fs::rename(dir.join("publishing"), dir.join("host.holder.json")).unwrap();
    finish(child(&dir, false));
    assert!(!dir.join("entered").exists());
    drop(witness);
    drop(host);
    finish(waiting);
    assert_eq!(
        std::fs::read_to_string(dir.join("entered")).unwrap(),
        "true"
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn an_independent_thread_cannot_share_the_owning_threads_admission() {
    let dir = std::env::temp_dir().join(format!("l2d-gate-thread-{}", std::process::id()));
    let guard = ResourceGuard::at(&dir, "thread-one", BusyChoice::Refuse, true).unwrap();
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let error =
                    ResourceGuard::at(&dir, "thread-two", BusyChoice::Refuse, false).unwrap_err();
                assert!(error.to_string().contains("thread-one"));
            })
            .join()
            .unwrap();
    });
    drop(guard);
    let guard = ResourceGuard::at(&dir, "after-thread", BusyChoice::Refuse, true).unwrap();
    assert!(guard.observation.host_holder.is_none());
    drop(guard);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
#[ignore = "requires L2D_TEST_PYTHON pointing at a local Python interpreter"]
fn python_training_and_rust_share_admission_and_crash_release() {
    let python = std::env::var("L2D_TEST_PYTHON").expect("local Python interpreter");
    let dir = std::env::temp_dir().join(format!("l2d-gate-python-{}", std::process::id()));
    let module = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/science");
    let run = |script: &str| {
        Command::new(&python)
            .args(["-c", script])
            .arg(&module)
            .arg(dir.join("host.lock"))
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .output()
            .unwrap()
    };
    let guard = ResourceGuard::at(&dir, "first-process", BusyChoice::Refuse, false).unwrap();
    let refused = run(
        "import sys; sys.path.insert(0,sys.argv[1]); import memguard; memguard.LOCK=sys.argv[2]; memguard.hungry('python training').__enter__()",
    );
    assert!(!refused.status.success());
    assert!(
        String::from_utf8_lossy(&refused.stderr).contains("first-process"),
        "{}",
        String::from_utf8_lossy(&refused.stderr)
    );
    drop(guard);
    let crash = run(
        "import sys,os; sys.path.insert(0,sys.argv[1]); import memguard; memguard.LOCK=sys.argv[2]; g=memguard.hungry('python training'); g.__enter__(); os._exit(64)",
    );
    assert_eq!(
        crash.status.code(),
        Some(64),
        "{}",
        String::from_utf8_lossy(&crash.stderr)
    );
    let guard = ResourceGuard::at(&dir, "after-python-crash", BusyChoice::Refuse, true).unwrap();
    assert!(guard.observation.host_holder.is_none());
    drop(guard);
    let mut waiting_command = Command::new(&python);
    let waiting = waiting_command.args(["-c", "import sys,json; sys.path.insert(0,sys.argv[1]); import memguard; memguard.LOCK=sys.argv[2]; g=memguard.hungry('python waiting',wait=True); g.__enter__(); print(json.dumps(g.report())); g.__exit__()"])
        .arg(&module).arg(dir.join("host.lock")).env("PYTHONDONTWRITEBYTECODE", "1")
        .stdout(Stdio::piped()).stderr(Stdio::piped());
    let guard = ResourceGuard::at(&dir, "first-process", BusyChoice::Refuse, false).unwrap();
    let mut waiter = waiting.spawn().unwrap();
    std::thread::sleep(Duration::from_millis(250));
    assert!(waiter.try_wait().unwrap().is_none());
    drop(guard);
    let output = waiter.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("\"waited_for_host\": true"));
    let mut owner = Command::new(&python).args(["-c", "import sys,time,pathlib; sys.path.insert(0,sys.argv[1]); import memguard; memguard.LOCK=sys.argv[2]; g=memguard.hungry('first-process Python'); g.__enter__(); root=pathlib.Path(sys.argv[2]).parent; (root/'python-ready').touch(); end=time.monotonic()+10\nwhile not (root/'python-release').exists() and time.monotonic()<end: time.sleep(.01)\ng.__exit__()"])
        .arg(&module).arg(dir.join("host.lock")).env("PYTHONDONTWRITEBYTECODE", "1")
        .stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    let started = Instant::now();
    while !dir.join("python-ready").exists() {
        assert!(started.elapsed() < Duration::from_secs(10));
        assert!(owner.try_wait().unwrap().is_none());
        std::thread::sleep(Duration::from_millis(10));
    }
    let error = ResourceGuard::at(&dir, "Rust contender", BusyChoice::Refuse, false).unwrap_err();
    assert!(error.to_string().contains("first-process Python"));
    std::fs::write(dir.join("python-release"), "").unwrap();
    finish(owner);
    let cleanup = run(
        "import sys; sys.path.insert(0,sys.argv[1]); import memguard; from pathlib import Path; memguard.LOCK=sys.argv[2]; g=memguard.hungry('cleanup failure'); g.__enter__(); original=Path.unlink\ndef fail(*a,**k): raise OSError('injected cleanup error')\nPath.unlink=fail\ntry: g.__exit__()\nexcept OSError: pass\nfinally: Path.unlink=original\nassert g.handle is None and g.witness is None\nwith memguard.hungry('after cleanup failure'): pass",
    );
    assert!(
        cleanup.status.success(),
        "{}",
        String::from_utf8_lossy(&cleanup.stderr)
    );
    std::fs::remove_dir_all(dir).unwrap();
}
