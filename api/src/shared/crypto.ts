import { webcrypto } from "crypto";

const subtle = webcrypto.subtle;

async function getAesKey(secret: string): Promise<CryptoKey> {
  const encoder = new TextEncoder();
  const keyMaterial = await subtle.digest("SHA-256", encoder.encode(secret));
  return subtle.importKey("raw", keyMaterial, { name: "AES-GCM" }, false, [
    "encrypt",
    "decrypt",
  ]);
}

export async function encryptValue(
  plaintext: string,
  secret: string
): Promise<string> {
  const key = await getAesKey(secret);
  const encoder = new TextEncoder();
  const iv = webcrypto.getRandomValues(new Uint8Array(12));
  const encrypted = await subtle.encrypt(
    { name: "AES-GCM", iv },
    key,
    encoder.encode(plaintext)
  );
  const ivB64 = Buffer.from(iv).toString("base64");
  const dataB64 = Buffer.from(encrypted).toString("base64");
  return `${ivB64}.${dataB64}`;
}

export async function decryptValue(
  encrypted: string,
  secret: string
): Promise<string> {
  const key = await getAesKey(secret);
  const [ivB64, dataB64] = encrypted.split(".");
  if (!ivB64 || !dataB64) throw new Error("Invalid encrypted format");
  const iv = Buffer.from(ivB64, "base64");
  const data = Buffer.from(dataB64, "base64");
  const decrypted = await subtle.decrypt({ name: "AES-GCM", iv }, key, data);
  return new TextDecoder().decode(decrypted);
}
