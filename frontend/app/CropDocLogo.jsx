"use client";

import React, { useEffect, useState } from "react";
import { useTheme } from "next-themes";

const CropDocLogo = () => {
  const { theme, systemTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <div className="flex items-center">
        <span className="text-4xl font-bold -top-3 relative">
          Crop
          <span className="absolute -bottom-4 left-0 text-4xl text-green-700">
            Doc
          </span>
        </span>
      </div>
    );
  }

  const currentTheme = theme === "system" ? systemTheme : theme;
  const isDark = currentTheme === "dark";

  return (
    <div className="flex items-center">
      <span className="text-4xl font-bold -top-3 relative">
        Crop
        <span
          className={`absolute -bottom-4 left-0 text-4xl ${
            isDark ? "text-green-300" : "text-green-700"
          }`}
        >
          Doc
        </span>
      </span>
    </div>
  );
};

export default CropDocLogo;