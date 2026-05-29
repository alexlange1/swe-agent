import { Hero } from "@/components/landing/Hero";
import {
  WhatWeTrack,
  Problem,
  HowItWorks,
  AgapScore,
  Features,
  Oracle,
  Alerts,
  Testimonials,
  FinalCta,
} from "@/components/landing/Sections";
import { Pricing } from "@/components/Pricing";

export default function HomePage() {
  return (
    <>
      <Hero />
      <WhatWeTrack />
      <Problem />
      <HowItWorks />
      <AgapScore />
      <Features />
      <Oracle />
      <Alerts />
      <Pricing />
      <Testimonials />
      <FinalCta />
    </>
  );
}
