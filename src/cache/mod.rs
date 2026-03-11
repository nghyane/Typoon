use anyhow::Result;
use sha2::{Digest, Sha256};

pub struct DiskCache {
    db: redb::Database,
}

const TRANSLATIONS: redb::TableDefinition<&str, &[u8]> = redb::TableDefinition::new("translations");

impl DiskCache {
    pub fn new(cache_dir: &str) -> Result<Self> {
        std::fs::create_dir_all(cache_dir)?;
        let db_path = format!("{cache_dir}/comic-scan.redb");
        let db = redb::Database::create(&db_path)?;

        // Ensure table exists
        let txn = db.begin_write()?;
        let _ = txn.open_table(TRANSLATIONS)?;
        txn.commit()?;

        Ok(Self { db })
    }

    /// Build cache key: hash(image_hash + target_lang + model)
    pub fn key(image_hash: &str, target_lang: &str, model: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(image_hash.as_bytes());
        hasher.update(target_lang.as_bytes());
        hasher.update(model.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    pub fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(TRANSLATIONS)?;
        Ok(table.get(key)?.map(|v| v.value().to_vec()))
    }

    pub fn set(&self, key: &str, value: &[u8]) -> Result<()> {
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(TRANSLATIONS)?;
            table.insert(key, value)?;
        }
        txn.commit()?;
        Ok(())
    }
}
