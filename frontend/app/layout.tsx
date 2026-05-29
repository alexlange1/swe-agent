import type { Metadata } from "next";
import "./globals.css";
import { Nav } from "@/components/Nav";
import { Footer } from "@/components/Footer";

export const metadata: Metadata = {
  title: "Alpha | Bittensor Subnet Intelligence",
  description:
    "Find the alpha gap before everyone else. AI-driven intelligence across every Bittensor subnet — dev activity, emissions, whale flows, and social velocity.",
  metadataBase: new URL("https://alpha.local"),
  openGraph: {
    title: "Alpha | Bittensor Subnet Intelligence",
    description: "Find the alpha gap before everyone else.",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">
        <Nav />
        <main>{children}</main>
        <Footer />
      </body>
    </html>
  );
}
