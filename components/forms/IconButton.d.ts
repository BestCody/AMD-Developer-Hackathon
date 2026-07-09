import * as React from "react";

export interface IconButtonProps {
  /** Glyph node — pair with the Lucide CDN icon set (see readme Iconography). */
  icon: React.ReactNode;
  /** Translucent gray chip over photography (default) vs solid ring button. */
  translucent?: boolean;
  size?: number;
  label: string;
  onClick?: () => void;
  style?: React.CSSProperties;
}

export function IconButton(props: IconButtonProps): JSX.Element;
