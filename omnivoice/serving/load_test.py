#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Small HTTP load test for comparing Ray Serve batching configurations."""

import argparse
import asyncio
import io
import statistics
import time
import wave
from dataclasses import dataclass

import aiohttp


@dataclass
class Result:
    latency: float
    audio_duration: float
    status: int
    error: str | None = None


async def _request(
    session: aiohttp.ClientSession, url: str, payload: dict
) -> Result:
    started = time.perf_counter()
    try:
        async with session.post(url, json=payload) as response:
            body = await response.read()
            latency = time.perf_counter() - started
            if response.status != 200:
                return Result(
                    latency,
                    0.0,
                    response.status,
                    body.decode(errors="replace"),
                )
            with wave.open(io.BytesIO(body), "rb") as wav:
                duration = wav.getnframes() / wav.getframerate()
            return Result(latency, duration, response.status)
    except Exception as error:
        return Result(time.perf_counter() - started, 0.0, 0, str(error))


async def _run(args) -> None:
    url = f"{args.base_url.rstrip('/')}/v1/audio/speech"
    payload = {
        "input": args.text,
        "language": args.language,
        "num_step": args.num_step,
    }
    if args.ref_voice:
        payload["ref_voice"] = args.ref_voice

    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    connector = aiohttp.TCPConnector(limit=max(args.concurrency, 1))
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for _ in range(args.warmup):
            result = await _request(session, url, payload)
            if result.error:
                raise RuntimeError(f"warmup failed ({result.status}): {result.error}")

        semaphore = asyncio.Semaphore(args.concurrency)

        async def limited_request():
            async with semaphore:
                return await _request(session, url, payload)

        started = time.perf_counter()
        results = await asyncio.gather(
            *(limited_request() for _ in range(args.requests))
        )
        elapsed = time.perf_counter() - started

    successes = [result for result in results if result.error is None]
    latencies = sorted(result.latency for result in successes)

    def percentile(p: float) -> float:
        if not latencies:
            return float("nan")
        index = round((len(latencies) - 1) * p)
        return latencies[index]

    audio_seconds = sum(result.audio_duration for result in successes)
    print(
        f"requests={args.requests} success={len(successes)} "
        f"errors={len(results) - len(successes)}"
    )
    print(f"elapsed_s={elapsed:.3f} requests_per_s={len(successes) / elapsed:.3f}")
    print(f"audio_seconds_per_s={audio_seconds / elapsed:.3f}")
    print(
        "latency_s "
        f"mean={statistics.fmean(latencies) if latencies else float('nan'):.3f} "
        f"p50={percentile(0.50):.3f} p95={percentile(0.95):.3f} "
        f"p99={percentile(0.99):.3f}"
    )
    for result in results:
        if result.error:
            print(f"error status={result.status}: {result.error[:300]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--text", default="Xin chào, đây là OmniVoice.")
    parser.add_argument("--language", default="Vietnamese")
    parser.add_argument("--ref-voice")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--num-step", type=int, default=32)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1 or args.warmup < 0:
        parser.error("requests/concurrency must be positive and warmup must be >= 0")
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
