const CACHE_NAME = 'emotion-detection-cache-v1';
const urlsToCache = [
  '/',
  'https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js'
];

self.addEventListener('install', function(event) {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(function(cache) {
        return cache.addAll(urlsToCache);
      })
  );
});

self.addEventListener('fetch', function(event) {
  event.respondWith(
    caches.match(event.request)
      .then(function(response) {
        if (response) {
          return response;
        }
        return fetch(event.request);
      })
  );
});
