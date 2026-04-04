/** AIFGE external links — shared between Header dropdown and Hub footer */
export const aifgeLinks = [
  { label: "Courses", href: "https://aiforglobaleducation.org/courses/" },
  { label: "Resources", href: "https://aiforglobaleducation.org/resources/" },
  { label: "Volunteering", href: "https://aiforglobaleducation.org/volunteering/" },
  { label: "About Us", href: "https://aiforglobaleducation.org/about-us/" },
] as const;

/** Internal nav links — shared between Header and mobile menu */
export const navLinks = [
  { label: "Home", to: "/" },
  { label: "Assess", to: "/assess" },
  { label: "Frameworks", to: "/frameworks" },
  { label: "Audit", to: "/audit" },
  { label: "Policy", to: "/policy" },
  { label: "Evaluate", to: "/evaluate" },
] as const;
