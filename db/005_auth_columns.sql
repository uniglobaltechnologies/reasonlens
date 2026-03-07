-- Add password hash column for built-in auth
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS password_hash TEXT;

-- Make email unique for auth lookups
CREATE UNIQUE INDEX IF NOT EXISTS idx_profiles_email ON profiles(email);
