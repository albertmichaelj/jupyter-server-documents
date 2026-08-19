// Copyright (c) Jupyter Development Team.
// Distributed under the terms of the Modified BSD License.

import { YNotebook } from '@jupyter/ydoc';
import { nullTranslator } from '@jupyterlab/translation';
import {
  acceptDialog,
  dismissDialog,
  FakeUserManager,
  sleep,
  waitForDialog
} from '@jupyterlab/testutils';
import { requestAPI } from '../docprovider/requests';
import {
  getConnectionEpoch,
  EPOCH_BUMP_MIN_DOWNTIME_MS
} from '../docprovider/executionChain';
import { WebSocketProvider } from '../docprovider/yprovider';

jest.mock('../docprovider/requests', () => ({
  requestAPI: jest.fn()
}));

interface IMockWsProvider {
  emit: (eventName: string, payload: any) => void;
  connect: jest.Mock;
  disconnect: jest.Mock;
}

jest.mock('y-websocket', () => ({
  WebsocketProvider: class {
    roomname: string;
    wsUnsuccessfulReconnects = 0;
    maxBackoffTime = 2500;
    connect = jest.fn();
    disconnect = jest.fn();
    messageHandlers: any[] = [];
    private _listeners = new Map<string, Set<(payload: any) => void>>();

    constructor(_url: string, roomname: string) {
      this.roomname = roomname;
    }

    on(eventName: string, listener: (payload: any) => void): void {
      if (!this._listeners.has(eventName)) {
        this._listeners.set(eventName, new Set());
      }
      this._listeners.get(eventName)!.add(listener);
    }

    off(eventName: string, listener: (payload: any) => void): void {
      this._listeners.get(eventName)?.delete(listener);
    }

    destroy(): void {
      this._listeners.clear();
    }

    emit(eventName: string, payload: any): void {
      this._listeners.get(eventName)?.forEach(listener => listener(payload));
    }
  }
}));

async function waitForProviderConnect(
  provider: WebSocketProvider
): Promise<IMockWsProvider> {
  for (let i = 0; i < 10; i++) {
    const wsProvider = provider.wsProvider as unknown as IMockWsProvider;
    if (wsProvider) {
      return wsProvider;
    }
    await Promise.resolve();
  }
  throw new Error('WebSocket provider was not initialized');
}

function createProvider(
  options: { path?: string; model?: YNotebook } = {}
): WebSocketProvider {
  const { path = 'test.ipynb', model = new YNotebook() } = options;
  const translator = nullTranslator.load('test');
  const identity = {
    username: 'Test User',
    display_name: 'Test User',
    name: 'Test User',
    initials: 'TU',
    color: 'blue'
  };
  const user = new FakeUserManager({}, identity, {});
  const mockApp = {
    shell: { widgets: jest.fn().mockReturnValue([]) }
  };

  return new WebSocketProvider({
    app: mockApp as any,
    path,
    contentType: 'notebook',
    format: 'json',
    model,
    user,
    translator,
    url: 'ws://localhost:8888/api/collaboration/room'
  });
}

describe('WebSocketProvider reconnection', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.spyOn(console, 'warn').mockImplementation();
    jest.spyOn(console, 'error').mockImplementation();
    jest.spyOn(console, 'info').mockImplementation();
    (requestAPI as jest.Mock).mockResolvedValue({ id: 'test-file-id' });
  });

  afterEach(async () => {
    // Clean up any stray dialogs left in the DOM
    await dismissDialog(undefined, 50).catch(() => {});
    document.body.innerHTML = '';
  });

  describe('_onConnectionClosed', () => {
    it('should log warning and not dispose for transient close code 1006', async () => {
      const model = new YNotebook();
      const disposeSpy = jest.spyOn(model, 'dispose');
      const provider = createProvider({ model });
      const wsProvider = await waitForProviderConnect(provider);

      wsProvider.emit('connection-close', { code: 1006, reason: '' });

      expect(console.warn).toHaveBeenCalledWith(
        expect.stringContaining('code=1006'),
        expect.anything()
      );
      expect(disposeSpy).not.toHaveBeenCalled();
      provider.dispose();
    });

    it('should log warning and not dispose for transient close code 1001', async () => {
      const model = new YNotebook();
      const disposeSpy = jest.spyOn(model, 'dispose');
      const provider = createProvider({ model });
      const wsProvider = await waitForProviderConnect(provider);

      wsProvider.emit('connection-close', { code: 1001, reason: '' });

      expect(console.warn).toHaveBeenCalledWith(
        expect.stringContaining('code=1001'),
        expect.anything()
      );
      expect(disposeSpy).not.toHaveBeenCalled();
      provider.dispose();
    });

    it('should dispose shared model for close code 4001 (out-of-band move)', async () => {
      const model = new YNotebook();
      const disposeSpy = jest.spyOn(model, 'dispose');
      const provider = createProvider({ model });
      const wsProvider = await waitForProviderConnect(provider);

      wsProvider.emit('connection-close', { code: 4001, reason: '' });

      expect(disposeSpy).toHaveBeenCalled();
      provider.dispose();
    });

    it('should dispose shared model for close code 4002 (in-band deletion)', async () => {
      const model = new YNotebook();
      const disposeSpy = jest.spyOn(model, 'dispose');
      const provider = createProvider({ model });
      const wsProvider = await waitForProviderConnect(provider);

      wsProvider.emit('connection-close', { code: 4002, reason: '' });

      expect(disposeSpy).toHaveBeenCalled();
      provider.dispose();
    });
  });

  describe('_onStatus reconnect tracking', () => {
    it('should reset reconnect attempts on successful connection', async () => {
      const provider = createProvider();
      const wsProvider = await waitForProviderConnect(provider);

      wsProvider.emit('status', { status: 'disconnected' });
      wsProvider.emit('status', { status: 'disconnected' });
      wsProvider.emit('status', { status: 'connected' });

      expect((provider as any)._reconnectAttempts).toBe(0);
      provider.dispose();
    });

    it('should not show retry dialog before MAX_RECONNECT_ATTEMPTS exceeded', async () => {
      const provider = createProvider();
      const wsProvider = await waitForProviderConnect(provider);

      // Emit exactly MAX_RECONNECT_ATTEMPTS disconnects (should not trigger dialog)
      for (let i = 0; i < WebSocketProvider.MAX_RECONNECT_ATTEMPTS; i++) {
        wsProvider.emit('status', { status: 'disconnected' });
      }

      await expect(waitForDialog(undefined, 100)).rejects.toThrow(
        'Dialog not found'
      );
      provider.dispose();
    });

    it('should log error and show retry dialog after MAX_RECONNECT_ATTEMPTS exceeded', async () => {
      const provider = createProvider();
      const wsProvider = await waitForProviderConnect(provider);

      for (let i = 0; i <= WebSocketProvider.MAX_RECONNECT_ATTEMPTS; i++) {
        wsProvider.emit('status', { status: 'disconnected' });
      }

      expect(console.error).toHaveBeenCalledWith(
        expect.stringContaining('failed to reconnect')
      );
      expect(wsProvider.disconnect).toHaveBeenCalled();
      await waitForDialog(undefined, 200);
      await dismissDialog(undefined, 200);
      provider.dispose();
    });

    it('should resume connection when user clicks Retry', async () => {
      const provider = createProvider();
      const wsProvider = await waitForProviderConnect(provider);

      for (let i = 0; i <= WebSocketProvider.MAX_RECONNECT_ATTEMPTS; i++) {
        wsProvider.emit('status', { status: 'disconnected' });
      }

      await waitForDialog(undefined, 200);
      await acceptDialog(undefined, 200);
      await sleep(50);

      expect(wsProvider.connect).toHaveBeenCalled();
      expect((provider as any)._reconnectAttempts).toBe(0);
      expect((WebSocketProvider as any)._reconnectedManually).toBe(true);
      provider.dispose();
    });

    it('should log success on reconnect after retry', async () => {
      const provider = createProvider();
      const wsProvider = await waitForProviderConnect(provider);

      // Trigger retry dialog
      for (let i = 0; i <= WebSocketProvider.MAX_RECONNECT_ATTEMPTS; i++) {
        wsProvider.emit('status', { status: 'disconnected' });
      }

      // Click Retry
      await waitForDialog(undefined, 200);
      await acceptDialog(undefined, 200);
      await sleep(50);

      // Connection succeeds
      wsProvider.emit('status', { status: 'connected' });

      expect(console.info).toHaveBeenCalledWith(
        'WebSocket reconnected successfully.'
      );
      provider.dispose();
    });
  });

  describe('_onStatus execution-chain epoch', () => {
    it('bumps the epoch only when downtime could have outlived the room', async () => {
      const provider = createProvider();
      const wsProvider = await waitForProviderConnect(provider);
      // roomName is `${format}:${contentType}:${fileId}` from _connect.
      const room = 'json:notebook:test-file-id';

      const nowSpy = jest.spyOn(Date, 'now');
      try {
        nowSpy.mockReturnValue(100_000);
        wsProvider.emit('status', { status: 'connected' });
        const base = getConnectionEpoch(room);

        // Short blip: the room is guaranteed alive, and the chain is the
        // FIFO protection for a request in flight across the blip — it must
        // survive.
        wsProvider.emit('status', { status: 'disconnected' });
        nowSpy.mockReturnValue(102_000);
        wsProvider.emit('status', { status: 'connected' });
        expect(getConnectionEpoch(room)).toBe(base);

        // Walk-away: downtime is measured against the last liveness stamp
        // (taken at the previous 'connected'), and retry cycles emitting
        // further 'disconnected' events must not refresh it. Past the gate
        // the chain must break.
        nowSpy.mockReturnValue(102_000);
        wsProvider.emit('status', { status: 'disconnected' });
        nowSpy.mockReturnValue(102_000 + EPOCH_BUMP_MIN_DOWNTIME_MS - 1_000);
        wsProvider.emit('status', { status: 'disconnected' });
        nowSpy.mockReturnValue(102_000 + EPOCH_BUMP_MIN_DOWNTIME_MS);
        wsProvider.emit('status', { status: 'connected' });
        expect(getConnectionEpoch(room)).toBe(base + 1);
      } finally {
        nowSpy.mockRestore();
      }
      provider.dispose();
    });

    it('bumps after a sleep even though the disconnect is only observed at wake', async () => {
      // During laptop sleep no JS runs: the 'disconnected' event for a
      // connection that died at sleep start is only DELIVERED at wake,
      // seconds before the reconnect completes. Measuring downtime from
      // that event made a 20-minute lid close look like a 3-second blip
      // and the epoch never bumped — the gate must instead measure from
      // the last time the connection was known alive.
      const provider = createProvider();
      const wsProvider = await waitForProviderConnect(provider);
      const room = 'json:notebook:test-file-id';

      const nowSpy = jest.spyOn(Date, 'now');
      try {
        nowSpy.mockReturnValue(500_000);
        wsProvider.emit('status', { status: 'connected' });
        const base = getConnectionEpoch(room);

        // Lid closes; JS freezes; nothing fires for 20 minutes. At wake,
        // the stale-connection close and the reconnect land back-to-back.
        const wake = 500_000 + 20 * 60_000;
        nowSpy.mockReturnValue(wake);
        wsProvider.emit('status', { status: 'disconnected' });
        nowSpy.mockReturnValue(wake + 3_000);
        wsProvider.emit('status', { status: 'connected' });
        expect(getConnectionEpoch(room)).toBe(base + 1);
      } finally {
        nowSpy.mockRestore();
      }
      provider.dispose();
    });
  });
});
