import * as React from "react";

export interface ButtonProps {
  /** Visual treatment. `primary` is the one true action signal (full pill, Action Blue). */
  variant?: "primary" | "secondary-pill" | "dark-utility" | "pearl-capsule" | "store-hero";
  disabled?: boolean;
  onClick?: () => void;
  children: React.ReactNode;
  style?: React.CSSProperties;
}

export function Button(props: ButtonProps): JSX.Element;
