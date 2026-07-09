import * as React from "react";

export interface InputProps {
  /** `pill` = search-input grammar (full pill, 44px). `default` = rectangular form field. */
  variant?: "default" | "pill";
  icon?: React.ReactNode;
  placeholder?: string;
  value?: string;
  onChange?: (e: React.ChangeEvent<HTMLInputElement>) => void;
  style?: React.CSSProperties;
}

export function Input(props: InputProps): JSX.Element;
