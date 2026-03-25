export const dynamic = 'force-dynamic';

export async function POST(req) {
  // Use API_BASE as the primary (runtime) variable, fallback to NEXT_PUBLIC_ for local dev.
  const apiBase = process.env.API_BASE || process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8080";
  const body = await req.text();

  const upstream = await fetch(`${apiBase}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body
  });

  const text = await upstream.text();
  return new Response(text, {
    status: upstream.status,
    headers: { "Content-Type": upstream.headers.get("content-type") || "application/json" }
  });
}

