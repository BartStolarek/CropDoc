"use client";

import React from 'react';
import { useTheme } from "next-themes"; // Import from next-themes

const CropDocLogo = () => {
  const { theme } = useTheme();
  const isDark = theme === 'dark'; // Check if the theme is dark

  return (
    <div className="flex items-center">
      <span 
        className={`text-4xl font-bold -top-3  relative`}
      >
        Crop
        <span 
          className={`absolute -bottom-4 left-0 text-4xl ${isDark ? 'text-green-300' : 'text-green-700'}`}
        >
          Doc
        </span>
      </span>
    </div>
  );
};

export default CropDocLogo;
