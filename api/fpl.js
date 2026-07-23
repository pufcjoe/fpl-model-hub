// Serverless proxy for the official FPL API — kills the browser CORS block.
const ALLOWED = /^https:\/\/fantasy\.premierleague\.com\/api\/[\w\-\/\?\=\&]*$/;

export default async function handler(req, res) {
  const url = req.query.url;
  if (!url || !ALLOWED.test(url)) {
    return res.status(400).json({ error: 'url must be a fantasy.premierleague.com/api endpoint' });
  }
  try {
    const r = await fetch(url, {
      headers: { 'User-Agent': 'Mozilla/5.0 (fpl-model-hub)' },
      cache: 'no-store',
    });
    if (!r.ok) return res.status(r.status).json({ error: 'upstream ' + r.status });
    const data = await r.json();
    res.setHeader('Cache-Control', 's-maxage=300, stale-while-revalidate=600');
    return res.status(200).json(data);
  } catch (e) {
    return res.status(502).json({ error: 'fetch failed' });
  }
}
