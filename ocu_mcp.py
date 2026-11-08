"""Persistent stdio Open Computer Use transport for the optional Python runner."""
import json
import os
import queue
import shlex
import signal
import subprocess
import threading
import time


class OcuMCP:
    def __init__(self, command=None, timeout=120):
        # SAWII_OCU_COMMAND overrides the whole argv (space-separated, shlex rules) -
        # used by tests and by anyone running a stand-in server. SAWII_OCU_EXECUTABLE
        # just names the executable and keeps the plain "mcp" argument.
        override = os.getenv('SAWII_OCU_COMMAND')
        self.command = command or (
            shlex.split(override) if override else
            (["cmd.exe","/D","/C","open-computer-use","mcp"] if os.name == 'nt' and not os.getenv('SAWII_OCU_EXECUTABLE') else [os.getenv('SAWII_OCU_EXECUTABLE', 'open-computer-use'), 'mcp']))
        self.timeout = timeout
        self.lock = threading.Lock()
        self.process = None
        self.serial = 0
        self.schemas = []

    def close(self):
        process, self.process = self.process, None
        if process:
            try:
                if os.name == 'nt':
                    subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                else:
                    os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.kill()
            process.wait(timeout=5)
            for stream in (process.stdin, process.stdout):
                if stream:
                    stream.close()

    def _start(self):
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL, text=True, encoding='utf-8', start_new_session=os.name != 'nt')
        process = self.process
        self.lines = queue.Queue()
        lines = self.lines
        def read():
            try:
                while True:
                    line = process.stdout.readline(16 * 1024 * 1024 + 1)
                    if not line or len(line) > 16 * 1024 * 1024:
                        break
                    lines.put(line)
            except (OSError, ValueError):
                pass
            finally:
                lines.put(None)
        threading.Thread(target=read, daemon=True).start()
        self._request('initialize', {'protocolVersion':'2025-03-26','capabilities':{},
                      'clientInfo':{'name':'sawii-companion','version':'1'}})
        self._send({'jsonrpc':'2.0','method':'notifications/initialized'})
        result = self._request('tools/list', {})
        self.schemas = []
        while True:
            self.schemas.extend(result['tools'])
            if not result.get('nextCursor'):
                break
            if len(self.schemas) > 128:
                raise ValueError('Too many OCU tools')
            result = self._request('tools/list', {'cursor':result['nextCursor']})

    def _send(self, obj):
        self.process.stdin.write(json.dumps(obj) + '\n')
        self.process.stdin.flush()

    def _request(self, method, params):
        self.serial += 1
        serial = self.serial
        self._send({'jsonrpc':'2.0','id':serial,'method':method,'params':params})
        deadline = time.monotonic() + self.timeout
        while True:
            line = self.lines.get(timeout=max(.01, deadline-time.monotonic()))
            if line is None:
                raise ConnectionError('OCU disconnected')
            value = json.loads(line)
            if value.get('id') != serial:
                continue
            if 'error' in value:
                raise RuntimeError(str(value['error']))
            return value['result']

    def tools(self):
        with self.lock:
            try:
                if self.process is None:
                    self._start()
                return [{'type':'function','function':{'name':'ocu_'+s['name'],
                         'description':s.get('description','Open Computer Use'),
                         'parameters':s['inputSchema']}} for s in self.schemas]
            except Exception:
                self.close()
                raise

    def call(self, name, args, stop=None):
        with self.lock:
            try:
                if stop is not None and stop.is_set():
                    raise RuntimeError("OCU call cancelled")
                if self.process is None:
                    self._start()
                if name not in {s['name'] for s in self.schemas}:
                    raise ValueError('Unknown OCU tool')
                return json.dumps(self._request('tools/call', {'name':name,'arguments':args}))
            except Exception:
                self.close()
                raise
