// Runs mongod on 127.0.0.1:27017 with a persistent dbPath so LibreChat users
// survive restarts. No Docker and no brew needed; mongodb-memory-server
// downloads the mongod binary on first run. Ctrl-C (or SIGTERM) stops it.
const fs = require('fs');
const path = require('path');
const { MongoMemoryServer } = require('mongodb-memory-server');

const dbPath = process.env.MONGO_DB_PATH || path.join(__dirname, '.mongo-data');
fs.mkdirSync(dbPath, { recursive: true });

(async () => {
  const mongod = await MongoMemoryServer.create({
    instance: { ip: '127.0.0.1', port: 27017, dbPath, storageEngine: 'wiredTiger' },
  });
  console.log(`mongod listening at ${mongod.getUri()} (dbPath ${dbPath})`);
  const stop = async () => {
    await mongod.stop({ doCleanup: false });
    process.exit(0);
  };
  process.on('SIGINT', stop);
  process.on('SIGTERM', stop);
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
