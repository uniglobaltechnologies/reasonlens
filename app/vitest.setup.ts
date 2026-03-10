// Provide a working localStorage for jsdom environment
// Node.js 25+ has a built-in localStorage that doesn't implement the full Storage API
const store: Record<string, string> = {};

const storage: Storage = {
  getItem: (key: string) => store[key] ?? null,
  setItem: (key: string, value: string) => { store[key] = String(value); },
  removeItem: (key: string) => { delete store[key]; },
  clear: () => { for (const key of Object.keys(store)) delete store[key]; },
  key: (index: number) => Object.keys(store)[index] ?? null,
  get length() { return Object.keys(store).length; },
};

Object.defineProperty(globalThis, "localStorage", { value: storage, writable: true });
