import * as React from "react";

export interface TagProps {
  selected?: boolean;
  children: React.ReactNode;
  onClick?: () => void;
  style?: React.CSSProperties;
}

export function Tag(props: TagProps): JSX.Element;
