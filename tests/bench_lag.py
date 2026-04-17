"""Detect event loop blocking from sync scan inside asyncio.create_task.

If scan blocks the loop, translate's async response processing is delayed.
"""

import asyncio
import time


async def main():
    print("=" * 60)
    print("Event Loop Blocking Test")
    print("=" * 60)

    loop = asyncio.get_event_loop()
    lag_samples: list[float] = []

    # Monitor: check event loop responsiveness every 10ms
    async def lag_monitor():
        while True:
            t0 = loop.time()
            await asyncio.sleep(0.01)  # expect ~10ms
            actual = (loop.time() - t0) * 1000
            lag_samples.append(actual - 10)  # excess over expected

    monitor = asyncio.create_task(lag_monitor())

    # Simulate: blocking scan (150ms) inside create_task
    async def fake_prescan():
        await asyncio.sleep(0)  # yield once
        time.sleep(0.15)  # BLOCKS event loop
        return "scanned"

    # Simulate: async translate (300ms network I/O)
    async def fake_translate():
        t0 = time.monotonic()
        await asyncio.sleep(0.30)
        actual = (time.monotonic() - t0) * 1000
        return actual

    print("\n▸ Test 1: scan in create_task + translate concurrently")
    lag_samples.clear()
    scan_task = asyncio.create_task(fake_prescan())
    translate_actual = await fake_translate()
    await scan_task
    monitor.cancel()

    max_lag = max(lag_samples) if lag_samples else 0
    print(f"  Translate expected: 300ms, actual: {translate_actual:.0f}ms")
    print(f"  Max event loop lag: {max_lag:.0f}ms")
    print(f"  Lag > 50ms samples: {sum(1 for l in lag_samples if l > 50)}")

    blocked = max_lag > 100
    print(f"\n  {'⚠ BLOCKED' if blocked else '✓ OK'}: event loop {'was' if blocked else 'not'} blocked")

    # Test 2: with run_in_executor
    print("\n▸ Test 2: scan in run_in_executor + translate concurrently")
    lag_samples.clear()
    monitor = asyncio.create_task(lag_monitor())

    def blocking_scan():
        time.sleep(0.15)
        return "scanned"

    t0 = time.monotonic()
    scan_future = loop.run_in_executor(None, blocking_scan)
    translate_actual2 = await fake_translate()
    await scan_future
    total = (time.monotonic() - t0) * 1000

    monitor.cancel()
    max_lag2 = max(lag_samples) if lag_samples else 0
    print(f"  Translate expected: 300ms, actual: {translate_actual2:.0f}ms")
    print(f"  Max event loop lag: {max_lag2:.0f}ms")
    print(f"  Total wall time: {total:.0f}ms (vs 300ms ideal)")

    blocked2 = max_lag2 > 100
    print(f"\n  {'⚠ BLOCKED' if blocked2 else '✓ OK'}: event loop {'was' if blocked2 else 'not'} blocked")

    # Summary
    print("\n" + "=" * 60)
    print("Conclusion:")
    if blocked and not blocked2:
        print("  create_task blocks loop → use run_in_executor for scan")
    elif not blocked:
        print("  No blocking detected (scan may be too fast to notice)")
    else:
        print("  Both approaches show lag (GIL contention)")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
